"""Run this script with 'torchrun'."""

import gzip
import logging
import sys
from pathlib import Path
from typing import Optional, TextIO

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from packaging import version
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

from olmo.config import CheckpointType, TrainConfig
from olmo.data import build_train_dataloader
from olmo.eval import build_evaluators
from olmo.exceptions import OLMoCliError, OLMoConfigurationError
from olmo.model import OLMo
from olmo.optim import BoltOnWarmupScheduler, build_optimizer, build_scheduler
from olmo.torch_util import (
    barrier,
    get_default_device,
    get_global_rank,
    get_local_rank,
    get_local_world_size,
    get_world_size,
    peak_gpu_memory,
    seed_all,
)
from olmo.trainer import Trainer
from olmo.util import clean_opt, log_extra_field, prepare_cli_environment, get_eval_intervals
import os
import subprocess

log = logging.getLogger("train")


def main(cfg: TrainConfig) -> None:
    # Ensure run name set.
    if cfg.run_name is None:
        raise OLMoConfigurationError("--run_name is required")
    # if we use different t_max for different batch sizes via specifying cfg.sweep_bs_steps
    if cfg.sweep and cfg.sweep_bs_steps is not None:
        global_train_batch_size, max_duration = cfg.sweep_bs_steps.split('-')
        cfg.global_train_batch_size = int(global_train_batch_size)
    # sweeps
    if cfg.sweep:
        cfg.run_name = "-".join(["bs"+str(cfg.global_train_batch_size), cfg.sweep_opt_momentum, "lr"+str(cfg.optimizer.learning_rate)])

    if 'schedule_free' in cfg.scheduler.name or 'cosine' in cfg.scheduler.name or 'wsd' in cfg.scheduler.name:
        cfg.optimizer.use_ewa = False

    if cfg.optimizer.hardcode_opt_params: # should be close to optimal
        ema_list = {64: 0.9995, 128: 0.9995, 256: 0.9995, 512: 0.9995, 1024: 0.999, 2048: 0.998, 4096: 0.995, 8192: 0.99, 16384: 0.99}
        beta2_list = {64: 0.999, 128: 0.999, 256: 0.995, 512: 0.99, 1024: 0.99, 2048: 0.95, 4096: 0.95, 8192: 0.95, 16384: 0.95}
        cfg.optimizer.ewa_decay = ema_list[cfg.global_train_batch_size]
        cfg.optimizer.beta2 = beta2_list[cfg.global_train_batch_size]
        log.info(f"Hardcode and switch to a beta2 {cfg.optimizer.beta2}, decay rate {cfg.optimizer.ewa_decay} in EMA for {cfg.global_train_batch_size} batch size")

    if cfg.optimizer.multiple_ewa:
        cfg.optimizer.ewa_decay = [.99, .9968, .999, .99968, .9999]
    if cfg.optimizer.use_ewa:
        cfg.run_name += f"-ewa{cfg.optimizer.ewa_decay}"
    log_extra_field("run_name", cfg.run_name)

    # Sanity check
    if (cfg.reset_optimizer_state or cfg.reset_trainer_state) and cfg.load_path is None:
        log.warning(
            "You want to reset the optimizer or trainer state, but we're not loading from the checkpoint. The"
            "setting has no effect."
        )
    
    try:
        result = subprocess.run(['sudo', 'chmod', '-R', 'a+rwx', str(cfg.save_folder)], capture_output=True, text=True)
        if result.returncode == 0:
            log.info(f"Permissions to path {str(cfg.save_folder)} changed successfully.")
        else:
            log.info(f"Failed to change permissions to path {str(cfg.save_folder)}")
            log.info(result.stderr)
    except Exception as e:
        log.info(f"Failed to change permissions to path {str(cfg.save_folder)}")
        log.info(e)

    # Maybe start W&B run.
    if cfg.wandb is not None and (get_global_rank() == 0 or not cfg.wandb.rank_zero_only):
        wandb_dir = Path(cfg.save_folder) / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        if cfg.sweep:
            wandb.init(
                dir=wandb_dir,
                project=cfg.wandb.project,
                # entity=cfg.wandb.entity,
                group=cfg.wandb.group,
                name=cfg.run_name, 
                tags=cfg.wandb.tags,
                config=cfg.asdict(exclude=["wandb"])
            )
        else:
            wandb.init(
                dir=wandb_dir,
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                group=cfg.wandb.group,
                name=cfg.run_name, 
                tags=cfg.wandb.tags,
                config=cfg.asdict(exclude=["wandb"])
            )
        os.environ['WANDB_RUN_ID'] = wandb.run.id
    barrier()
    cfg.save_folder = Path(cfg.save_folder) / cfg.run_name / str(os.environ['WANDB_RUN_ID'])

    if cfg.sweep:
        optimizer_name, momentum = cfg.sweep_opt_momentum.split('_')
        # Construct optimizer and learning rate scheduler.
        log.info(cfg.sweep_opt_momentum)
        if optimizer_name not in ['adam', 'adamw', 'sgd']:
            raise ValueError(f"Optimizer {optimizer_name} not used for sweeping experiments")
        momentum = float(momentum)
        
        if 'schedule_free' in cfg.scheduler.name:
            cfg.optimizer.name = 'adamw_schedule_free'
            cfg.optimizer.use_ewa = False
        else:
            cfg.optimizer.name = optimizer_name

        cfg.optimizer.momentum = momentum
        if cfg.optimizer.name in ['adam', 'adamw', 'adamw_schedule_free']:
            cfg.optimizer.betas = (momentum, cfg.optimizer.beta2)

        if cfg.sweep_bs_steps is not None:
            global_train_batch_size, max_duration = cfg.sweep_bs_steps.split('-')
            cfg.global_train_batch_size = int(global_train_batch_size)
            cfg.max_duration = int(max_duration)
    cfg.scheduler.t_warmup = int(cfg.scheduler.warmup_ratio * (cfg.chinchilla_token/cfg.global_train_batch_size))
    cfg.scheduler.t_stable = int((1-cfg.scheduler.decay_ratio) * cfg.max_duration) - cfg.scheduler.t_warmup
    assert cfg.scheduler.t_warmup > 0
    log.info(f"Scheduler warmup steps: {cfg.scheduler.t_warmup}/{cfg.max_duration}")
    if cfg.scheduler.name in ['wsd']:
        log.info(f"LR Scheduler begins decaying at the {cfg.scheduler.t_stable}-th/{cfg.max_duration} step")

    eval_intervals = get_eval_intervals(cfg)
    
    log.info(f"Wandb project: {cfg.wandb.project}, Wandb group: {cfg.wandb.group}, Wandb tags: {cfg.wandb.tags}")
    log.info(f"Run name: {cfg.run_name}")
    log.info(f"Sweep or not? {cfg.sweep}")
    log.info(cfg.sweep_opt_momentum)
    log.info(f"global_train_batch_size: {cfg.global_train_batch_size}")
    log.info(f"learning_rate: {cfg.optimizer.learning_rate}")

    barrier()

    # Set CUDA device.
    torch.cuda.set_device(f"cuda:{get_local_rank()}")
    device = torch.device("cuda")

    # Fill some configuration options.
    cfg.model.precision = cfg.precision
    # NOTE for sweeping experiments
    if cfg.device_train_microbatch_size > cfg.global_train_batch_size:
        cfg.device_train_microbatch_size = cfg.global_train_batch_size
    # NOTE add for ctx length ablation to avoid OOM
    cfg.device_eval_batch_size = cfg.device_train_microbatch_size
    cfg.eval_subset_num_batches = max(100, int(64 / cfg.device_eval_batch_size * 50)) # 256 / bs # 100
    log.info(f"Eval subset num batches: {cfg.eval_subset_num_batches}")
    for eval_cfg in cfg.evaluators:
        eval_cfg.subset_num_batches = cfg.eval_subset_num_batches
    
    cfg.device_train_batch_size = cfg.global_train_batch_size // get_world_size()
    assert cfg.device_train_batch_size is not None  # for mypy
    cfg.device_train_grad_accum = cfg.device_train_batch_size // cfg.device_train_microbatch_size
    if cfg.optimizer.no_decay_norm_and_bias is not None:
        log.warning(
            "You set the deprecated config option `no_decay_norm_and_bias`. For compatibility, this"
            "setting will take precedence over all other weight decay configurations. Please change"
            "your config to use `decay_norm_and_bias` and `decay_embeddings` instead."
        )
        cfg.optimizer.decay_norm_and_bias = not cfg.optimizer.no_decay_norm_and_bias
        cfg.optimizer.decay_embeddings = not cfg.optimizer.no_decay_norm_and_bias
        cfg.optimizer.no_decay_norm_and_bias = None  # So nobody uses this by accident.

    barrier()

    # Set seed.
    seed_all(cfg.seed)

    # Construct data loader.
    train_loader = build_train_dataloader(cfg)

    # Construct evaluators.
    evaluators = build_evaluators(cfg, device)
    barrier()

    # Initialize the model.
    log.info("Building model...")
    olmo_model = OLMo(cfg.model)
    log.info(f"Total number of parameters: {olmo_model.num_params():,d}")
    log.info(f"Number of non-embedding parameters: {olmo_model.num_params(include_embedding=False):,d}")
    log.info(f"Peak GPU Memory (MB) before FSDP: {int(peak_gpu_memory() or 0)}")

    olmo_model.set_activation_checkpointing(cfg.activation_checkpointing)
    
    # Wrap the model in FSDP.
    log.info("Wrapping model with FDSP...")
    wrap_policy = olmo_model.get_fsdp_wrap_policy(cfg.fsdp.wrapping_strategy)

    if version.parse(torch.__version__) >= version.parse("2.1.0"):
        # This prevents any parameters from being initialized twice
        def dummy_init_fn(module: torch.nn.Module) -> None:
            module.to_empty(device=get_default_device())

        param_init_fn = dummy_init_fn
    else:
        param_init_fn = None

    # Set up device mesh for hybrid sharding in order to specify which nodes are assoicated to a given model replica
    device_mesh = None
    hybrid_sharding_fsdp_kwargs = {}
    if cfg.fsdp.sharding_strategy in (ShardingStrategy.HYBRID_SHARD, ShardingStrategy._HYBRID_SHARD_ZERO2):
        if version.parse(torch.__version__) < version.parse("2.2.0"):
            # Device mesh was not added to PyTorch until v2.2.0
            raise OLMoConfigurationError(
                "OLMo training does not correctly support hybrid sharding before torch 2.2.0"
            )

        from torch.distributed.device_mesh import init_device_mesh

        num_model_replicas = cfg.fsdp.hybrid_sharding_num_model_replicas or (
            get_world_size() // get_local_world_size()
        )

        if num_model_replicas <= 0:
            raise OLMoConfigurationError("fsdp.hybrid_sharding_num_model_replicas must be a positive integer")

        num_nodes = get_world_size() // get_local_world_size()
        if num_nodes > 1 and num_nodes % num_model_replicas != 0:
            raise OLMoConfigurationError("fsdp.hybrid_sharding_num_model_replicas must divide number of nodes")

        device_mesh = init_device_mesh("cuda", (num_model_replicas, get_world_size() // num_model_replicas))
        hybrid_sharding_fsdp_kwargs["device_mesh"] = device_mesh

    fsdp_model = FSDP(
        olmo_model,
        sharding_strategy=cfg.fsdp.sharding_strategy,
        mixed_precision=cfg.fsdp_precision,
        auto_wrap_policy=wrap_policy,
        use_orig_params=cfg.fsdp.use_orig_params,  # needed for compile and some of our optimizer/parameter metrics
        limit_all_gathers=True,
        device_id=get_local_rank(),
        param_init_fn=param_init_fn,
        **hybrid_sharding_fsdp_kwargs,
    )
    # when param_init_fn is None, FSDP will call reset_parameters() automatically
    if param_init_fn is not None:
        olmo_model.reset_parameters()

    log.info(f"Peak GPU Memory (MB) after FSDP: {int(peak_gpu_memory() or 0)}")
    log.info("Model:")
    log.info(fsdp_model)

    # Construct optimizer and learning rate scheduler.
    optim = build_optimizer(cfg, fsdp_model)
    log.info("Optimizer:")
    log.info(optim)
    if 'schedule_free' in cfg.optimizer.name:
        scheduler = None
        log.info("No scheduler used")
    else:
        scheduler = build_scheduler(cfg)
    log.info("Scheduler:")
    log.info(scheduler)

    # Data indices file.
    indices_file: Optional[TextIO] = None
    if cfg.save_data_indices:
        indices_file_path = Path(cfg.save_folder) / f"data-indices/rank{get_global_rank()}.tsv.gz"
        if indices_file_path.exists() and not cfg.save_overwrite:
            raise OLMoConfigurationError(f"{indices_file_path} already exists, use --save_overwrite to overwrite")
        indices_file_path.parent.mkdir(exist_ok=True, parents=True)
        indices_file = gzip.open(indices_file_path, "wt")

    # Consolidate components into `Trainer` object.
    with Trainer(
        cfg=cfg,
        epoch=cfg.epoch,
        model=olmo_model,
        fsdp_model=fsdp_model,
        optim=optim,
        scheduler=scheduler,
        train_loader=train_loader,
        device=device,
        evaluators=evaluators,
        indices_file=indices_file,
        eval_intervals=eval_intervals
    ) as trainer:
        if not cfg.dry_run and not cfg.no_pre_train_checkpoint and cfg.load_path is None:
            checkpoint_type = (
                CheckpointType.sharded if cfg.save_num_checkpoints_to_keep != 0 else CheckpointType.unsharded
            )

            # We save a checkpoint up-front to make sure this won't fail (due to disk space or whatever).
            # log.info("Saving pre-train checkpoint...")
            # checkpoint_path, local_checkpoint_cache = trainer.save_checkpoint(checkpoint_type=checkpoint_type)
            # log.info(f"Checkpoint saved to {checkpoint_path}")

            # # And they we verify that we can load it.
            # log.info("Attempting to load pre-train checkpoint...")
            # trainer.restore_checkpoint(
            #     checkpoint_path, checkpoint_type=checkpoint_type, local_cache=local_checkpoint_cache
            # )
            # log.info("Checkpoint successfully loaded")

        # if cfg.load_path is not None:
        #     log.info(f"Loading checkpoint from {cfg.load_path}...")
        #     trainer.restore_checkpoint(
        #         cfg.load_path,
        #         load_optimizer_state=not cfg.reset_optimizer_state,
        #         load_trainer_state=not cfg.reset_trainer_state,
        #         sharded_checkpointer=cfg.load_path_sharded_checkpointer,
        #     )
        #     log.info("Checkpoint successfully loaded")

        #     # If we have to, set a new scheduler:
        #     if cfg.reset_optimizer_state and not cfg.reset_trainer_state:
        #         trainer.scheduler = BoltOnWarmupScheduler.wrap(
        #             trainer.scheduler,
        #             trainer.global_step,
        #             int(trainer.global_step + cfg.scheduler.t_warmup),
        #         )

        # if cfg.force_save_unsharded:
        #     log.info("Saving unsharded checkpoint...")
        #     checkpoint_path, _ = trainer.save_checkpoint(checkpoint_type=CheckpointType.unsharded)
        #     log.info(f"Unsharded checkpoint saved to {checkpoint_path}")

        if cfg.compile is not None:
            # TODO (epwalsh): trying to compile the whole train step results in a compile-time error from within
            # the optimizer. We should investigate this further at some point.
            trainer.train_batch = torch.compile(trainer.train_batch, **cfg.compile.asdict())  # type: ignore
            log.info("Torch.compiled the train_batch.")
            # TODO (epwalsh): compiling the `eval_batch()` method is a little sketchy since the inputs will look
            # different for different eval tasks. That might be okay, but it might not be.
            # trainer.eval_batch = torch.compile(trainer.eval_batch, **cfg.compile.asdict())  # type: ignore
            # log.info("Torch.compiled the eval_batch.")

        if not cfg.dry_run:
            log.info("Starting training...")
            trainer.fit()
            log.info("Training complete")
        else:
            log.info("Dry run complete")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError as e:
        print(f"failed to set multiprocessing start method: {e}")
    log.info(f"Multiprocessing start method set to '{mp.get_start_method()}'")

    # Initialize process group.
    dist.init_process_group(backend="nccl")
    log.info("Process group initialized")

    prepare_cli_environment()
    log.info("CLI environment prepared")

    try:
        yaml_path, args_list = sys.argv[1], sys.argv[2:]
    except IndexError:
        raise OLMoCliError(f"Usage: {sys.argv[0]} [CONFIG_PATH] [OPTIONS]")

    cfg = TrainConfig.load(yaml_path, [clean_opt(s) for s in args_list])
    main(cfg)
