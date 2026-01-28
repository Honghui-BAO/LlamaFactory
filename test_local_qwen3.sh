#!/bin/bash

# 本地代码路径（相对于当前目录）
LOCAL_MODEL_CODE="model/qwen3"
# 权重路径（远端原始权重）
WEIGHTS_PATH="/llm-reco-ssd-share/zhangzixing/onerec_pretrain/model_output/pro/sft/8b_v0.1.0_fromstg2_noamazon/step6500/global_step6500/converted"
# 输出路径
OUTPUT_DIR="/llm-reco-ssd-share/baohonghui/torchrec/SumRec/merge/ckpt/test_local_load_v2"

export WANDB_DISABLED=true

# 🚨 准备工作：
# 为了让 transformers 能在这个目录下找到完整的模型，我们需要链接权重文件
echo "Checking weights and tokenizer files..."
for file in $(ls $WEIGHTS_PATH); do
    if [[ ! -f "$LOCAL_MODEL_CODE/$file" ]] && [[ ! -L "$LOCAL_MODEL_CODE/$file" ]]; then
        ln -s "$WEIGHTS_PATH/$file" "$LOCAL_MODEL_CODE/$file"
    fi
done

echo "Starting training task with local code in $LOCAL_MODEL_CODE..."

deepspeed --num_gpus 8 \
    src/train.py \
    --deepspeed examples/deepspeed/ds_z2_config.json \
    --stage sft \
    --model_name_or_path $LOCAL_MODEL_CODE \
    --do_train \
    --dataset meta2sid_dataset,rec_dataset \
    --template qwen3 \
    --finetuning_type full \
    --output_dir $OUTPUT_DIR \
    --overwrite_cache \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --num_train_epochs 1.0 \
    --bf16 \
    --trust_remote_code True \
    --flash_attn fa2

echo "Test task finished."
