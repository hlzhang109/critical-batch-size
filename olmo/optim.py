import logging
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, replace
from math import cos, pi, sqrt
from typing import Any, Dict, List, Optional, Tuple
import math

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.optim.optimizer import Optimizer as OptimizerBase

from . import LayerNormBase
from .config import OptimizerType, SchedulerConfig, SchedulerType, TrainConfig
from .torch_util import get_default_device, is_distributed
from itertools import cycle

__all__ = [
    "Optimizer",
    "LionW",
    "AdamW",
    "Scheduler",
    "CosWithWarmup",
    "LinearWithWarmup",
    "InvSqrtWithWarmup",
    "WarmupStableDecay",
    "MaxScheduler",
    "ConstantWithWarmupScheduler",
    "BoltOnWarmupScheduler",
    "build_optimizer",
    "build_scheduler",
]


log = logging.getLogger(__name__)


class Optimizer(OptimizerBase):
    def _clean_param_name(self, name: str) -> str:
        return name.replace("_fsdp_wrapped_module.", "")

    @torch.no_grad()
    def clip_grads_and_collect_metrics(
        self,
        global_step: int,
        collect_param_metrics: bool = True,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Clips gradients for every group that has the field `max_grad_norm`.
        At the same time collect metrics for each parameter and its gradient.
        """
        device = get_default_device()

        # NOTE (epwalsh): during distributed training we're making an assumption that the order of
        # the param groups and the params within each group are the same across all ranks.
        # This is justified since we initialize the parameter groups in every rank by iterating over
        # `module.parameters()` or `module.named_modules()` / `module.named_parameters()`, each of which
        # provides a consistent order.
        #  For each parameter (with a gradient) we'll collect:
        # - min, max, avg, norm of the param itself
        # - min, max, avg, norm of the param's gradient
        # - min, max, avg, norm of any additional per-parameter optimizer state metrics returned from
        #   `self.get_state_for_param()`.
        # Afterwards we'll reduce these all over all ranks.
        per_param_min_metrics: List[torch.Tensor] = []
        per_param_max_metrics: List[torch.Tensor] = []
        per_param_sum_metrics: List[torch.Tensor] = []
        per_param_norm_metrics: List[torch.Tensor] = []
        per_param_numel_metrics: List[torch.Tensor] = []

        per_param_min_metric_names: List[str] = []
        per_param_max_metric_names: List[str] = []
        per_param_avg_metric_names: List[str] = []
        per_param_norm_metric_names: List[str] = []

        dst_rank = 0
        if process_group is not None:
            dst_rank = dist.get_global_rank(process_group, 0)

        # Collect metrics locally.
        for group in self.param_groups:
            if is_distributed():
                assert group.get("sharded", True) is True

            for name, p in zip(group["param_names"], group["params"]):
                name = self._clean_param_name(name)
                # Always need to collect the norm of gradients for clipping, even if we're not collecting
                # other metrics.
                tensors: List[Optional[torch.Tensor]] = [p.grad]
                prefixes: List[str] = [f"grad/{name}"]
                if collect_param_metrics:
                    state = self.get_state_for_param(p)
                    sorted_state_keys = sorted([k for k in state.keys()])
                    tensors.extend([p] + [state[key] for key in sorted_state_keys])
                    prefixes.extend([f"param/{name}"] + [f"{key}/{name}" for key in sorted_state_keys])
                assert len(tensors) == len(prefixes)

                # Get min, max, avg, and norm for all `tensors` associated with the parameter.
                for x, prefix in zip(tensors, prefixes):
                    # grad or state tensors could be none for params that have their shards completely on
                    # other ranks.
                    if x is not None and x.numel() > 0:
                        if collect_param_metrics:
                            x_abs = x.abs()
                            per_param_min_metrics.append(x_abs.min().unsqueeze(0).to(dtype=torch.float32))
                            per_param_max_metrics.append(x_abs.max().unsqueeze(0).to(dtype=torch.float32))
                            per_param_sum_metrics.append(x.sum().unsqueeze(0).to(dtype=torch.float32))
                            per_param_numel_metrics.append(
                                torch.tensor([x.numel()], device=device, dtype=torch.float32)
                            )
                        per_param_norm_metrics.append(
                            torch.linalg.vector_norm(x, 2.0, dtype=torch.float32).unsqueeze(0)
                        )
                    else:
                        if collect_param_metrics:
                            per_param_min_metrics.append(
                                torch.tensor([float("inf")], device=device, dtype=torch.float32)
                            )
                            per_param_max_metrics.append(torch.tensor([0.0], device=device, dtype=torch.float32))
                            per_param_sum_metrics.append(torch.tensor([0.0], device=device, dtype=torch.float32))
                            per_param_numel_metrics.append(torch.tensor([0.0], device=device, dtype=torch.float32))
                        per_param_norm_metrics.append(torch.tensor([0.0], device=device, dtype=torch.float32))
                    if collect_param_metrics:
                        per_param_min_metric_names.append(f"{prefix}.min")
                        per_param_max_metric_names.append(f"{prefix}.max")
                        per_param_avg_metric_names.append(f"{prefix}.avg")
                    per_param_norm_metric_names.append(f"{prefix}.norm")

        assert (
            len(per_param_min_metrics)
            == len(per_param_min_metric_names)
            == len(per_param_max_metrics)
            == len(per_param_max_metric_names)
            == len(per_param_sum_metrics)
            == len(per_param_numel_metrics)
            == len(per_param_avg_metric_names)
        )
        assert len(per_param_norm_metrics) == len(per_param_norm_metric_names)

        def is_grad_norm_metric(metric_name: str) -> bool:
            return metric_name.startswith("grad/") and metric_name.endswith(".norm")

        # Now reduce metrics over all ranks.
        total_grad_norm: torch.Tensor
        per_param_avg_metrics: List[torch.Tensor] = []
        if is_distributed(): 
            # Reduce metrics across all ranks. Note that we can use a `reduce` for most cases
            # instead of an `all_reduce`, but we need `all_reduce` for norms so that all ranks
            # get the right value for gradient norms so they can clip correctly.
            # Reduce mins.
            if per_param_min_metrics:
                all_mins = torch.cat(per_param_min_metrics).to(device)
                dist.reduce(all_mins, dst_rank, op=dist.ReduceOp.MIN, group=process_group)
                per_param_min_metrics = all_mins.split(1)
            # Reduce maxs.
            if per_param_max_metrics:
                all_maxs = torch.cat(per_param_max_metrics).to(device)
                dist.reduce(all_maxs, dst_rank, op=dist.ReduceOp.MAX, group=process_group)
                per_param_max_metrics = all_maxs.split(1)
            # Reduce sums or just norms.
            all_norms = torch.cat(per_param_norm_metrics).to(device) ** 2.0
            if per_param_sum_metrics and per_param_numel_metrics:
                all_sums = torch.cat(per_param_sum_metrics).to(device)
                all_numels = torch.cat(per_param_numel_metrics).to(device)
                all_sums_norms_numels = torch.cat(
                    [all_sums.unsqueeze(0), all_norms.unsqueeze(0), all_numels.unsqueeze(0)], dim=0
                )
                dist.all_reduce(all_sums_norms_numels, op=dist.ReduceOp.SUM, group=process_group)
                all_sums, all_norms, all_numels = all_sums_norms_numels.split(1)
                # Get averages.
                # NOTE: could get infs for non-rank0 processes but that's okay.
                per_param_avg_metrics = (all_sums / all_numels).squeeze(0).split(1)
            else:
                dist.all_reduce(all_norms, op=dist.ReduceOp.SUM, group=process_group)
            grad_norm_metric_mask = torch.tensor(
                [float(is_grad_norm_metric(n)) for n in per_param_norm_metric_names], device=all_norms.device
            )
            total_grad_norm = (all_norms * grad_norm_metric_mask).sum() ** 0.5
            per_param_norm_metrics = (all_norms ** (0.5)).squeeze(0).split(1)
        else:
            total_grad_norm = (
                torch.cat(
                    [
                        m
                        for m, n in zip(per_param_norm_metrics, per_param_norm_metric_names)
                        if is_grad_norm_metric(n)
                    ]
                )
                ** 2.0
            ).sum() ** 0.5
            per_param_avg_metrics = [x / n for x, n in zip(per_param_sum_metrics, per_param_numel_metrics)]

        assert len(per_param_avg_metrics) == len(per_param_avg_metric_names)

        # Collect all metrics into a single dict.
        all_metrics: Dict[str, torch.Tensor] = {}
        for metric_name, metric in zip(per_param_min_metric_names, per_param_min_metrics):
            all_metrics[metric_name] = metric.squeeze(0)
        for metric_name, metric in zip(per_param_max_metric_names, per_param_max_metrics):
            all_metrics[metric_name] = metric.squeeze(0)
        for metric_name, metric in zip(per_param_avg_metric_names, per_param_avg_metrics):
            all_metrics[metric_name] = metric.squeeze(0)
        for metric_name, metric in zip(per_param_norm_metric_names, per_param_norm_metrics):
            all_metrics[metric_name] = metric.squeeze(0)
        all_metrics["total_grad_norm"] = total_grad_norm

        # Clip gradients.
        num_grads_clipped = 0
        num_eligible_grads = 0
        for group in self.param_groups:
            if (max_norm_ratio := group.get("max_grad_norm_ratio")) is not None:
                num_clipped = self._do_adaptive_clipping(
                    group, max_norm_ratio, global_step, all_metrics, collect_param_metrics=collect_param_metrics
                )
            elif (max_norm := group.get("max_grad_norm")) is not None:
                num_clipped = self._do_global_fixed_clipping(
                    group, max_norm, all_metrics, collect_param_metrics=collect_param_metrics
                )
            else:
                # No clipping needed.
                continue
            num_eligible_grads += len(group["params"])
            if num_clipped is not None:
                num_grads_clipped += num_clipped

        if collect_param_metrics:
            if num_eligible_grads > 0:
                clipping_rate = torch.tensor(num_grads_clipped / num_eligible_grads, device="cpu")
            else:
                clipping_rate = torch.tensor(0.0, device="cpu")
            all_metrics["clipping_rate"] = clipping_rate
            return all_metrics
        else:
            return {}

    @torch.no_grad()
    def _do_adaptive_clipping(
        self,
        group: Dict[str, Any],
        max_norm_ratio: float,
        global_step: int,
        all_metrics: Dict[str, torch.Tensor],
        collect_param_metrics: bool = True,
    ) -> Optional[int]:
        """
        Do adaptive gradient clipping on a param group.

        If ``collect_param_metrics`` is ``True`` this will return the total number of gradients clipped.
        """
        device = get_default_device()
        num_grads_clipped = 0
        beta1, beta2 = group["betas"]
        beta = max(beta1, beta2)
        for name, p in zip(group["param_names"], group["params"]):
            name = self._clean_param_name(name)
            grad_norm = all_metrics.get(f"grad/{name}.norm")
            if grad_norm is None:
                continue

            # Get or initialize the exponential average of grad norm.
            # TODO: The way we have it right now, every rank tracks the `grad_norm_exp_avg` of every parameter,
            # even parameters for which the corresponding local shard is empty. This has the potential to
            # cause some issues with the optimizer, as we ran into with https://github.com/allenai/LLM/pull/372.
            # So we should consider changing how we do this at some point so that we don't add any state
            # to parameters for which the local shard is empty. That would probably add extra distributed
            # communication, at least on steps where we have to log (i.e. when `collect_param_metrics=True`).
            state = self.state[p]
            grad_norm_exp_avg = state.get("grad_norm_exp_avg")
            if grad_norm_exp_avg is None:
                grad_norm_exp_avg = grad_norm.clone().to(device)
                # We don't want to add anything to `state` until `state` has been initialized, otherwise
                # this will crash some optimizers which rely on checking `len(state)`. The downside here
                # is that we won't start tracking `grad_norm_exp_avg` until the 2nd training step.
                if global_step > 1:
                    state["grad_norm_exp_avg"] = grad_norm_exp_avg

            max_allowed_norm = max_norm_ratio * grad_norm_exp_avg
            clip_coef = max_allowed_norm / (grad_norm + 1e-6)

            # Clip the gradients and update the exponential average.
            # Note that multiplying by the clamped coefficient is meaningless when it is
            # equal to 1, but it avoids the host-device sync that would result from `if clip_coef_clamped < 1`.
            clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
            if p.grad is not None:
                # p.grad could be none for some ranks when using FSDP.
                p.grad.detach().mul_(clip_coef_clamped.to(p.grad.device, p.grad.dtype))

            # Update the exponential average of the norm of the gradient with the clipped norm of the gradient.
            grad_norm_exp_avg.lerp_((grad_norm * clip_coef_clamped).to(grad_norm_exp_avg.device), 1 - beta)
            # Alternative: update with the *unclipped* norm of the gradient.
            #  grad_norm_exp_avg.lerp_(grad_norm.to(grad_norm_exp_avg.device), 1 - beta)

            if collect_param_metrics:
                # Can't avoid host-device sync here.
                if clip_coef_clamped < 1.0:
                    num_grads_clipped += 1
                all_metrics[f"grad_norm_exp_avg/{name}"] = grad_norm_exp_avg
        return num_grads_clipped if collect_param_metrics else None

    @torch.no_grad()
    def _do_global_fixed_clipping(
        self,
        group: Dict[str, Any],
        max_norm: float,
        all_metrics: Dict[str, torch.Tensor],
        collect_param_metrics: bool = True,
    ) -> Optional[int]:
        """
        Do global fixed gradient clipping on a param group.

        If ``collect_param_metrics`` is ``True`` this will return the total number of gradients clipped.
        """
        device = get_default_device()
        total_grad_norm = all_metrics["total_grad_norm"]
        clip_coef = max_norm / (total_grad_norm.to(device) + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        num_grads_clipped: Optional[int] = None
        if collect_param_metrics:
            # Can't avoid host-device sync here.
            if clip_coef_clamped < 1.0:
                num_grads_clipped = len(group["params"])
        for p in group["params"]:
            # Clip the gradients.
            # Note that multiplying by the clamped coefficient is meaningless when it is
            # equal to 1, but it avoids the host-device sync that would result from `if clip_coef_clamped < 1`.
            if p.grad is not None:
                # p.grad could be none for some ranks when using FSDP.
                p.grad.detach().mul_(clip_coef_clamped.to(p.grad.device, p.grad.dtype))
        return num_grads_clipped

    def get_post_step_metrics(
        self, module: nn.Module, process_group: Optional[dist.ProcessGroup] = None
    ) -> Dict[str, torch.Tensor]:
        del module, process_group
        return {}

    def get_state_for_param(self, param: nn.Parameter) -> Dict[str, Optional[torch.Tensor]]:
        del param
        return {}


class LionW(Optimizer):
    """
    Adapted from https://github.com/google/automl/blob/master/lion/lion_pytorch.py
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        assert lr > 0.0
        assert all([0.0 <= beta <= 1.0 for beta in betas])
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
        for group in self.param_groups:
            group["initial_lr"] = group["lr"]
        self._update_total_dot_prod: Optional[torch.Tensor] = None
        self._update_total_norm: Optional[torch.Tensor] = None
        self._signed_update_total_norm: Optional[torch.Tensor] = None

    def get_post_step_metrics(
        self, module: nn.Module, process_group: Optional[dist.ProcessGroup] = None
    ) -> Dict[str, torch.Tensor]:
        update_total_dot_prod = self._update_total_dot_prod
        update_total_norm = self._update_total_norm
        signed_update_total_norm = self._signed_update_total_norm
        if update_total_dot_prod is None or update_total_norm is None or signed_update_total_norm is None:
            return {}

        if is_distributed() and isinstance(module, FullyShardedDataParallel):
            # Reduce total dot prod and norms across all ranks.
            update_total_norm = update_total_norm**2.0
            signed_update_total_norm = signed_update_total_norm**2.0
            # Reduce all together to avoid multiple communication calls.
            all_together = torch.stack([update_total_dot_prod, update_total_norm, signed_update_total_norm])
            # Only need the final result on rank0, since that's where we log from.
            dist.reduce(
                all_together,
                0 if process_group is None else dist.get_global_rank(process_group, 0),
                group=process_group,
            )
            update_total_dot_prod, update_total_norm, signed_update_total_norm = all_together
            update_total_norm = update_total_norm**0.5
            signed_update_total_norm = signed_update_total_norm**0.5

        update_cos_sim = update_total_dot_prod / torch.max(
            update_total_norm * signed_update_total_norm, torch.tensor(1e-8, device=get_default_device())
        )
        return {"update_cos_sim": update_cos_sim}

    @torch.no_grad()
    def step(self, closure=None) -> None:
        if closure is not None:
            with torch.enable_grad():
                closure()

        update_total_dot_prod = torch.tensor(0.0, dtype=torch.float32)
        update_norms = []
        signed_update_norms = []

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Perform step weight decay
                p.data.mul_(1 - group["lr"] * group["weight_decay"])

                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]

                # Weight update
                update = exp_avg * beta1 + grad * (1 - beta1)
                signed_update = torch.sign(update)
                p.add_(signed_update, alpha=-group["lr"])

                # Decay the momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

                # Track dot product and norms of update vs signed update in order to calculate
                # their cosine similarity.
                update_total_dot_prod = update_total_dot_prod.to(update.device)
                update_total_dot_prod += torch.tensordot(update, signed_update, dims=len(update.shape))
                update_norms.append(torch.linalg.vector_norm(update, 2.0, dtype=torch.float32))
                signed_update_norms.append(torch.linalg.vector_norm(signed_update, 2.0, dtype=torch.float32))

        # Compute cosine similarity between update and signed update.
        self._update_total_dot_prod = update_total_dot_prod.to(get_default_device())
        self._update_total_norm = torch.linalg.vector_norm(
            torch.stack(update_norms),
            2.0,
            dtype=torch.float32,
        ).to(get_default_device())
        self._signed_update_total_norm = torch.linalg.vector_norm(
            torch.stack(signed_update_norms),
            2.0,
            dtype=torch.float32,
        ).to(get_default_device())

class AdamW(Optimizer):
    r"""Implements AdamW algorithm.

    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, use_w_avg=False,
                 w_avg_tau=0.998):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, train_mode=True)
        super(AdamW, self).__init__(params, defaults)
        self.use_w_avg = use_w_avg
        if type(w_avg_tau) == float:
            self.w_avg_tau_ls = [w_avg_tau]
        elif type(w_avg_tau) == list:
            self.w_avg_tau_ls = w_avg_tau
        assert type(self.w_avg_tau_ls) == list
        self.w_avg_tau = self.w_avg_tau_ls[0]
        self.w_avg_taus = cycle(self.w_avg_tau_ls) #w_avg_tau

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
    
    def eval(self):
        if self.use_w_avg:
            for group in self.param_groups:
                train_mode = group['train_mode']
                self.w_avg_tau = next(self.w_avg_taus)
                tau = self.w_avg_tau
                if train_mode:
                    for p in group['params']:
                        state = self.state[p]
                        if self.w_avg_tau == self.w_avg_tau_ls[0]: # [.99, .9968, .999, .99968, .9999]
                            state['temp'] = torch.clone(p.data) # save the current weights
                        bias_correction = 1 - tau**state['step']
                        p.data.lerp_(end=state[f'w_avg_{tau}'], weight=1.0).mul_(1/bias_correction)
                    if self.w_avg_tau == self.w_avg_tau_ls[-1]:
                        group['train_mode'] = False

    def train(self):
        if self.use_w_avg:
            for group in self.param_groups:
                train_mode = group['train_mode']
                if not train_mode:
                    for p in group['params']:
                        # tau = self.w_avg_tau
                        state = self.state[p]
                        if 'temp' in state: # _{tau} 
                            p.data.lerp_(end=state['temp'], weight=1.0)
                            del(state['temp'])
                    group['train_mode'] = True

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.mul_(1 - group['lr'] * group['weight_decay'])

                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                amsgrad = group['amsgrad']

                state = self.state[p]

                beta1, beta2 = group['betas']
                tau = self.w_avg_tau
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if self.use_w_avg:
                        for tau in self.w_avg_tau_ls:
                            state[f'w_avg_{tau}'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)
                if self.use_w_avg:
                    for tau in self.w_avg_tau_ls:
                        state[f'w_avg_{tau}'].mul_(tau).add_(p.data, alpha=1-tau)
                # w_avg.mul_(tau).add_(p.data, alpha=1-tau) # w_avg = state[f'w_avg_{tau}']

        return loss
    
    def get_state_for_param(self, param: nn.Parameter) -> Dict[str, Optional[torch.Tensor]]:
        return {key: self.state[param].get(key) for key in ("exp_avg", "exp_avg_sq")}  # type: ignore

# class AdamW(torch.optim.AdamW, Optimizer):
#     def get_state_for_param(self, param: nn.Parameter) -> Dict[str, Optional[torch.Tensor]]:
#         return {key: self.state[param].get(key) for key in ("exp_avg", "exp_avg_sq")}  # type: ignore

class SGD(torch.optim.SGD, Optimizer):
    def get_state_for_param(self, param: nn.Parameter) -> Dict[str, Optional[torch.Tensor]]:
        return {key: self.state[param].get(key) for key in ("momentum_buffer",)}  # type: ignore

class AdamWScheduleFree(Optimizer): #torch.optim.Optimizer
    r"""
    Schedule-Free AdamW
    As the name suggests, no scheduler is needed with this optimizer. 
    To add warmup, rather than using a learning rate schedule you can just
    set the warmup_steps parameter.
    
    This optimizer requires that .train() and .eval() be called before the
    beginning of training and evaluation respectively.
    
    Arguments:
        params (iterable): 
            Iterable of parameters to optimize or dicts defining 
            parameter groups.
        lr (float): 
            Learning rate parameter (default 0.0025)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999)).
        eps (float): 
            Term added to the denominator outside of the root operation to 
            improve numerical stability. (default: 1e-8).
        weight_decay (float): 
            Weight decay, i.e. a L2 penalty (default: 0).
        warmup_steps (int): Enables a linear learning rate warmup (default 0).
        r (float): Use polynomial weighting in the average 
            with power r (default 0).
        weight_lr_power (float): During warmup, the weights in the average will
            be equal to lr raised to this power. Set to 0 for no weighting
            (default 2.0).
        foreach (bool): Use a foreach-backed implementation of the optimizer.
            Should be significantly faster, but will have higher peak memory
            usage (default True).
    """
    def __init__(self,
                 params, 
                 lr=0.0025, 
                 betas=(0.9, 0.999), 
                 eps=1e-8,
                 weight_decay=0,
                 warmup_steps=0,
                 r=0.0,
                 weight_lr_power=2.0,
                 foreach=True
                 ):

        defaults = dict(lr=lr, 
                        betas=betas, 
                        eps=eps,
                        r=r,
                        k=0,
                        warmup_steps=warmup_steps,
                        train_mode=True,
                        weight_sum=0.0,
                        lr_max=-1.0,
                        weight_lr_power=weight_lr_power,
                        weight_decay=weight_decay,
                        foreach=foreach)
        super().__init__(params, defaults)
    
    def eval(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            beta1, _ = group['betas']
            if train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Set p.data to x
                        p.data.lerp_(end=state['z'], weight=1-1/beta1)
                group['train_mode'] = False

    def train(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            beta1, _ = group['betas']
            if not train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Set p.data to y
                        p.data.lerp_(end=state['z'], weight=1-beta1)
                group['train_mode'] = True

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            eps = group['eps']
            beta1, beta2 = group['betas']
            decay = group['weight_decay']
            k = group['k']
            r = group['r']
            warmup_steps = group['warmup_steps']
            weight_lr_power = group['weight_lr_power']
            
            if k < warmup_steps:
              sched = (k+1) / warmup_steps
            else:
              sched = 1.0
            
            bias_correction2 = 1 - beta2 ** (k+1)
            lr = group['lr']*sched*math.sqrt(bias_correction2)
            
            lr_max = group['lr_max'] = max(lr, group['lr_max'])
            
            weight = ((k+1)**r) * (lr_max**weight_lr_power)
            weight_sum = group['weight_sum'] = group['weight_sum'] + weight

            try:
                ckp1 = weight/weight_sum
            except ZeroDivisionError:
                ckp1 = 0

            if not group['train_mode']:
                raise Exception("Not in train mode!")

            active_p = [p for p in group['params'] if p.grad is not None]
            
            for p in active_p:
                if 'z' not in self.state[p]:
                    self.state[p]['z'] = torch.clone(p.data)
                    self.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)

            if group['foreach'] and len(active_p) > 0:
                y, grad, exp_avg_sq, z = zip(*[(p.data, 
                                                p.grad, 
                                                self.state[p]['exp_avg_sq'], 
                                                self.state[p]['z']) 
                                                for p in active_p])

                # Decay the first and second moment running average coefficient
                torch._foreach_mul_(exp_avg_sq, beta2)
                torch._foreach_addcmul_(exp_avg_sq, grad, grad, value=1-beta2)
                denom = torch._foreach_sqrt(exp_avg_sq)
                torch._foreach_add_(denom, eps)

                # Normalize grad in-place for memory efficiency
                torch._foreach_div_(grad, denom)

                # Weight decay calculated at y
                if decay != 0:
                    torch._foreach_add_(grad, y, alpha=decay)

                # These operations update y in-place,
                # without computing x explicitly.
                torch._foreach_lerp_(y, z, weight=ckp1)
                torch._foreach_add_(y, grad, alpha=lr*(beta1*(1-ckp1)-1))

                # z step
                torch._foreach_sub_(z, grad, alpha=lr)
            else:
                for p in active_p:
                    y = p.data # Notation to match theory
                    grad = p.grad.data

                    state = self.state[p]

                    z = state['z']
                    exp_avg_sq = state['exp_avg_sq']

                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                    denom = exp_avg_sq.sqrt().add_(eps)

                    # Reuse grad buffer for memory efficiency
                    grad_normalized = grad.div_(denom)

                    # Weight decay calculated at y
                    if decay != 0:
                        grad_normalized.add_(y, alpha=decay)

                    # These operations update y in-place,
                    # without computing x explicitly.
                    y.lerp_(end=z, weight=ckp1)
                    y.add_(grad_normalized, alpha=lr*(beta1*(1-ckp1)-1))

                    # z step
                    z.sub_(grad_normalized, alpha=lr)

            group['k'] = k+1
        return loss

    def get_state_for_param(self, param: nn.Parameter) -> Dict[str, Optional[torch.Tensor]]:
        return {key: self.state[param].get(key) for key in ("exp_avg", "exp_avg_sq")}  # type: ignore

@dataclass
class Scheduler(metaclass=ABCMeta):
    # NOTE: these fields are not given default values because otherwise dataclasses complains
    # about how the scheduler subclasses are defined.
    grad_clip_warmup_steps: Optional[int]
    grad_clip_warmup_factor: Optional[float]
    warmup_min_lr: Optional[float]

    @abstractmethod
    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
        raise NotImplementedError

    def _get_max_grad_norm_coeff(
        self, initial_value: Optional[float], step: int, max_steps: int
    ) -> Optional[float]:
        del max_steps  # might need this in the future, but for now I just wanted to match the API of `get_lr()`.
        if initial_value is None:
            return None
        elif (
            self.grad_clip_warmup_steps is None
            or self.grad_clip_warmup_factor is None
            or step > self.grad_clip_warmup_steps
        ):
            return initial_value
        else:
            return self.grad_clip_warmup_factor * initial_value

    def get_max_grad_norm(
        self, initial_max_grad_norm: Optional[float], step: int, max_steps: int
    ) -> Optional[float]:
        return self._get_max_grad_norm_coeff(initial_max_grad_norm, step, max_steps)

    def get_max_grad_norm_ratio(
        self, initial_max_grad_norm_ratio: Optional[float], step: int, max_steps: int
    ) -> Optional[float]:
        return self._get_max_grad_norm_coeff(initial_max_grad_norm_ratio, step, max_steps)

    def _linear_warmup(self, initial_lr: float, step: int, warmup_steps: int = 2000) -> float:
        warmup_min_lr = self.warmup_min_lr if self.warmup_min_lr is not None else initial_lr * 0.10
        assert 0 <= warmup_min_lr < initial_lr
        return warmup_min_lr + (initial_lr - warmup_min_lr) * min(step, warmup_steps) / warmup_steps


@dataclass
class CosWithWarmup(Scheduler):
    warmup_steps: int
    alpha_f: float = 0.1
    t_max: Optional[int] = None

    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
        max_steps = max_steps if self.t_max is None else self.t_max
        eta_min = initial_lr * self.alpha_f
        if step < self.warmup_steps:
            return self._linear_warmup(initial_lr, step, self.warmup_steps)
        elif step >= max_steps:
            return eta_min
        else:
            step = step - self.warmup_steps
            max_steps = max_steps - self.warmup_steps
            return eta_min + (initial_lr - eta_min) * (1 + cos(pi * step / max_steps)) / 2


@dataclass
class LinearWithWarmup(Scheduler):
    warmup_steps: int
    alpha_f: float = 0.1
    t_max: Optional[int] = None

    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
        max_steps = max_steps if self.t_max is None else self.t_max
        eta_min = initial_lr * self.alpha_f
        if step < self.warmup_steps:
            return self._linear_warmup(initial_lr, step, self.warmup_steps)
        elif step >= max_steps:
            return eta_min
        else:
            step = step - self.warmup_steps
            max_steps = max_steps - self.warmup_steps
            return initial_lr - (initial_lr - eta_min) * (step / max_steps)


@dataclass
class InvSqrtWithWarmup(Scheduler):
    warmup_steps: int

    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
        if step < self.warmup_steps:
            return self._linear_warmup(initial_lr, step, self.warmup_steps)
        del max_steps
        return initial_lr * sqrt(self.warmup_steps / max(self.warmup_steps, step))


@dataclass
class MaxScheduler(Scheduler):
    sched1: Scheduler
    sched2: Scheduler

    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
        return max(
            self.sched1.get_lr(initial_lr, step, max_steps), self.sched2.get_lr(initial_lr, step, max_steps)
        )


@dataclass
class BoltOnWarmupScheduler(Scheduler):
    inner: Scheduler
    warmup_start: int
    warmup_end: int

    @classmethod
    def wrap(cls, scheduler: Scheduler, warmup_start: int, warmup_end: int) -> "BoltOnWarmupScheduler":
        return cls(
            grad_clip_warmup_steps=None,
            grad_clip_warmup_factor=None,
            inner=scheduler,
            warmup_start=warmup_start,
            warmup_end=warmup_end,
            warmup_min_lr=None,
        )

    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
        if step < self.warmup_start:
            return 0.0
        if step < self.warmup_end:
            lr_at_intercept = self.inner.get_lr(initial_lr, self.warmup_end, max_steps)
            return lr_at_intercept * (step - self.warmup_start) / (self.warmup_end - self.warmup_start)
        else:
            return self.inner.get_lr(initial_lr, step, max_steps)

    def _get_max_grad_norm_coeff(
        self, initial_value: Optional[float], step: int, max_steps: int
    ) -> Optional[float]:
        return self.inner._get_max_grad_norm_coeff(initial_value, step, max_steps)


@dataclass
class ConstantWithWarmupScheduler(Scheduler):
    warmup_steps: int
    alpha_f: float = 0.1
    t_max: Optional[int] = None

    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
        max_steps = max_steps if self.t_max is None else self.t_max
        eta_min = initial_lr * self.alpha_f
        if step < self.warmup_steps:
            return self._linear_warmup(initial_lr, step, self.warmup_steps)
        elif step >= max_steps:
            return eta_min
        else:
            return self._linear_warmup(initial_lr, self.warmup_steps, self.warmup_steps)

@dataclass
class WarmupStableDecay(Scheduler):
    warmup_steps: int
    stable_steps: int
    alpha_f: float = 0.1
    t_max: Optional[int] = None

    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
        max_steps = max_steps if self.t_max is None else self.t_max
        eta_min = initial_lr * self.alpha_f
        if step < self.warmup_steps:
            return self._linear_warmup(initial_lr, step, self.warmup_steps)
        if step < self.stable_steps + self.warmup_steps:
            return initial_lr
        if step < max_steps:
            decay_factor = (step - self.stable_steps - self.warmup_steps) / (max_steps - self.stable_steps - self.warmup_steps)
            return initial_lr * (eta_min/initial_lr)**decay_factor # $f(s-T)=(lr_min/constant_lr)^{(s-S) / T} * constant_lr
        return eta_min

PARAM_GROUP_FIELDS = ("sharded", "max_grad_norm", "max_grad_norm_ratio", "param_names")


def get_param_groups(cfg: TrainConfig, model: nn.Module) -> List[Dict[str, Any]]:
    """
    Separate parameters into weight decay and non weight decay groups.
    """
    param_groups: List[Dict[str, Any]]
    param_group_defaults = {
        "sharded": isinstance(model, FullyShardedDataParallel),
        "max_grad_norm": cfg.max_grad_norm,
        "max_grad_norm_ratio": cfg.max_grad_norm_ratio,
    }

    # Separate out parameters that we don't want to apply weight decay to, like norms and biases.
    decay = set()
    no_decay = set()
    all_params = {}
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            # NOTE: because named_modules and named_parameters are recursive
            # we will see the same tensors p many many times, but doing it this way
            # allows us to know which parent module any tensor p belongs to...
            if not p.requires_grad:
                continue

            fpn = f"{mn}.{pn}" if mn else pn
            all_params[fpn] = p

            if pn.endswith("bias"):
                if cfg.optimizer.decay_norm_and_bias:
                    decay.add(fpn)
                else:
                    no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, nn.Linear):
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, (LayerNormBase, nn.LayerNorm)):
                if cfg.optimizer.decay_norm_and_bias:
                    decay.add(fpn)
                else:
                    no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, nn.Embedding):
                if cfg.optimizer.decay_embeddings:
                    decay.add(fpn)
                else:
                    no_decay.add(fpn)

    # Validate that we've considered every parameter
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f"parameters {inter_params} made it into both decay/no_decay sets!"
    assert (
        len(all_params.keys() - union_params) == 0
    ), f"parameters {all_params.keys() - union_params} were not separated into either decay/no_decay set!"

    # Create the pytorch optimizer groups.
    decay_sorted = sorted(list(decay))
    no_decay_sorted = sorted(list(no_decay))
    param_groups = []
    if len(decay_sorted) > 0:
        param_groups.append(
            {
                "params": [all_params[pn] for pn in decay_sorted],
                "param_names": decay_sorted,
                **param_group_defaults,
            }
        )
    if len(no_decay_sorted) > 0:
        param_groups.append(
            {
                "params": [all_params[pn] for pn in no_decay_sorted],
                "param_names": no_decay_sorted,
                "weight_decay": 0.0,
                **param_group_defaults,
            }
        )

    # Validate fields.
    for group in param_groups:
        for key in PARAM_GROUP_FIELDS:
            assert key in group

    return param_groups


def fix_optim_state_dict(optimizer: Optimizer, state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make sure old optim state dicts are compatible with new versions.
    """
    if len(state_dict["param_groups"]) == 1 and len(optimizer.param_groups) == 2:
        assert optimizer.param_groups[1]["weight_decay"] == 0.0

        # Decay
        decay_param_group = {k: v for k, v in state_dict["param_groups"][0].items() if k != "params"}
        decay_param_group["params"] = optimizer.state_dict()["param_groups"][0]["params"]

        # No decay.
        no_decay_param_group = {k: v for k, v in state_dict["param_groups"][0].items() if k != "params"}
        no_decay_param_group["weight_decay"] = 0.0
        no_decay_param_group["params"] = optimizer.state_dict()["param_groups"][1]["params"]

        state_dict["param_groups"] = [decay_param_group, no_decay_param_group]

    assert len(optimizer.param_groups) == len(state_dict["param_groups"])

    # Make sure:
    #  - All required fields are included in the state dict,
    #  - And that the values of those fields doesn't change from what's currently set in the optimizer,
    #    since we might have changed those fields on purpose after a restart.
    for group, sd_group in zip(optimizer.param_groups, state_dict["param_groups"]):
        for key in PARAM_GROUP_FIELDS:
            sd_group[key] = group[key]

    return state_dict


def build_optimizer(cfg: TrainConfig, model: nn.Module) -> Optimizer:
    param_groups = get_param_groups(cfg, model)
    log.info(f"Constructing optimizer with {len(param_groups)} param groups")
    if cfg.optimizer.name == OptimizerType.lionw:
        return LionW(
            param_groups,
            lr=cfg.optimizer.learning_rate,
            betas=cfg.optimizer.betas,
            weight_decay=cfg.optimizer.weight_decay,
        )
    elif cfg.optimizer.name == OptimizerType.adamw:
        return AdamW(
            param_groups,
            lr=cfg.optimizer.learning_rate,
            betas=cfg.optimizer.betas,
            weight_decay=cfg.optimizer.weight_decay,
            eps=cfg.optimizer.eps,
            use_w_avg=cfg.optimizer.use_ewa,
            w_avg_tau=cfg.optimizer.ewa_decay,
        )
    elif cfg.optimizer.name == OptimizerType.adamw_schedule_free:
        return AdamWScheduleFree(
            param_groups,
            lr=cfg.optimizer.learning_rate,
            betas=cfg.optimizer.betas, # default is (0.9, 0.999)
            weight_decay=cfg.optimizer.weight_decay,
            warmup_steps=cfg.scheduler.t_warmup,
            # r=cfg.optimizer.r, # r (float): Use polynomial weighting in the average with power r (default 0).
            # weight_lr_power=cfg.optimizer.weight_lr_power,  # weight_lr_power (float): During warmup, the weights in the average will be equal to lr raised to this power. Set to 0 for no weighting (default 2.0).
        )
    else:
        raise NotImplementedError


def build_scheduler(cfg: TrainConfig, sched_cfg: Optional[SchedulerConfig] = None) -> Scheduler:
    sched_cfg = sched_cfg if sched_cfg is not None else cfg.scheduler
    if sched_cfg.name == SchedulerType.cosine_with_warmup:
        return CosWithWarmup(
            grad_clip_warmup_steps=None
            if sched_cfg.grad_clip_warmup_steps is None
            else int(sched_cfg.grad_clip_warmup_steps),
            grad_clip_warmup_factor=sched_cfg.grad_clip_warmup_factor,
            warmup_steps=int(sched_cfg.t_warmup),
            alpha_f=sched_cfg.alpha_f,
            t_max=None if sched_cfg.t_max is None else int(sched_cfg.t_max),
            warmup_min_lr=sched_cfg.warmup_min_lr,
        )
    elif sched_cfg.name == SchedulerType.linear_with_warmup:
        return LinearWithWarmup(
            grad_clip_warmup_steps=None
            if sched_cfg.grad_clip_warmup_steps is None
            else int(sched_cfg.grad_clip_warmup_steps),
            grad_clip_warmup_factor=sched_cfg.grad_clip_warmup_factor,
            warmup_steps=int(sched_cfg.t_warmup),
            alpha_f=sched_cfg.alpha_f,
            t_max=None if sched_cfg.t_max is None else int(sched_cfg.t_max),
            warmup_min_lr=sched_cfg.warmup_min_lr,
        )
    elif sched_cfg.name == SchedulerType.inverse_sqrt_with_warmup:
        return InvSqrtWithWarmup(
            grad_clip_warmup_steps=None
            if sched_cfg.grad_clip_warmup_steps is None
            else int(sched_cfg.grad_clip_warmup_steps),
            grad_clip_warmup_factor=sched_cfg.grad_clip_warmup_factor,
            warmup_steps=int(sched_cfg.t_warmup),
            warmup_min_lr=sched_cfg.warmup_min_lr,
        )
    elif sched_cfg.name == SchedulerType.max_scheduler:
        return MaxScheduler(
            grad_clip_warmup_steps=None
            if sched_cfg.grad_clip_warmup_steps is None
            else int(sched_cfg.grad_clip_warmup_steps),
            grad_clip_warmup_factor=sched_cfg.grad_clip_warmup_factor,
            sched1=build_scheduler(cfg, replace(sched_cfg, name=SchedulerType.cosine_with_warmup)),
            sched2=build_scheduler(cfg, replace(sched_cfg, name=SchedulerType.inverse_sqrt_with_warmup)),
            warmup_min_lr=sched_cfg.warmup_min_lr,
        )
    elif sched_cfg.name == SchedulerType.constant:
        return ConstantWithWarmupScheduler(
            grad_clip_warmup_steps=None
            if sched_cfg.grad_clip_warmup_steps is None
            else int(sched_cfg.grad_clip_warmup_steps),
            grad_clip_warmup_factor=sched_cfg.grad_clip_warmup_factor,
            warmup_min_lr=sched_cfg.warmup_min_lr,
            warmup_steps=int(sched_cfg.t_warmup),
        )
    elif sched_cfg.name == SchedulerType.wsd:
        return WarmupStableDecay(
            grad_clip_warmup_steps=None
            if sched_cfg.grad_clip_warmup_steps is None
            else int(sched_cfg.grad_clip_warmup_steps),
            grad_clip_warmup_factor=sched_cfg.grad_clip_warmup_factor,
            warmup_steps=int(sched_cfg.t_warmup),
            stable_steps=int(sched_cfg.t_stable),
            alpha_f=sched_cfg.alpha_f,
            t_max=None if sched_cfg.t_max is None else int(sched_cfg.t_max),
            warmup_min_lr=sched_cfg.warmup_min_lr,
        )
    else:
        raise NotImplementedError