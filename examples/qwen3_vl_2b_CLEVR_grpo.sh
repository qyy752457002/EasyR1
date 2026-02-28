#!/bin/bash

set -x

# MODEL_PATH=Qwen/Qwen3-VL-2B-Instruct  # replace it with your local file path
MODEL_PATH=Qwen/Qwen3-VL-2B-Instruct
 
python3 -m verl.trainer.main \
    config=./examples/clevr_config.yaml \
    data.train_files=./examples/CLEVR/data/train-00000-of-00010.parquet  \
    data.val_files=./examples/CLEVR/data/test-00000-of-00002.parquet \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen3_vl_2b_CLEVR_grpo \
    trainer.n_gpus_per_node=4 \
    data.max_pixels=1258291 \
    data.max_prompt_length=3072 \
    data.max_response_length=1024 \
    data.val_batch_size=32