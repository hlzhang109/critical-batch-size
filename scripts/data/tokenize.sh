dataset=c4
data_dir=data

dolma tokens \
    --documents ${data_dir}/${dataset}/* \
    --tokenizer_name_or_path allenai/eleuther-ai-gpt-neox-20b-pii-special \
    --destination s3://${DATA_PATH}/olmo/preprocessed/gpt-neox-20b/${dataset} \
    --processes 64 \
    --ring_size 8 \
    --batch_size 10000 \
    --seed 0 \