#!/bin/bash

# Configuration
HOSTFILE="/etc/mpi/hostfile"
MASTER_ADDR=$(hostname -I | awk '{print $1}')
MASTER_PORT=12345

# Paths (Updated per user request)
model_path=/llm-reco-ssd-share/zhangzixing/onerec_pretrain/model_output/pro/sft/8b_v0.1.0_fromstg2_noamazon/step6500/global_step6500/converted
output_dir=/llm-reco-ssd-share/baohonghui/Reference/torchrec/SumRec/merge/ckpt/rmdr_v15_0_single_avg_mpi_v2

# Print Configuration
echo "--------------------------------------"
echo "LlamaFactory MPI Training Launch (128 Ranks)"
echo "--------------------------------------"
echo "MASTER: $MASTER_ADDR:$MASTER_PORT"
echo "OUTPUT: $output_dir"
echo "--------------------------------------"

# Multi-Node Launch using mpirun
mpirun --allow-run-as-root \
    --hostfile $HOSTFILE \
    -x http_proxy -x https_proxy -x no_proxy \
    -x MASTER_ADDR=$MASTER_ADDR -x MASTER_PORT=$MASTER_PORT \
    -x NCCL_DEBUG=INFO \
    -x NCCL_SOCKET_IFNAME=eth01 \
    -x NCCL_IB_HCA=mlx5 \
    -x NCCL_IB_GID_INDEX=3 \
    -x TORCH_NCCL_ENABLE_MONITORING=1 \
    -x TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600 \
    -x TORCH_DISTRIBUTED_DEBUG=DETAIL \
    -x TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    -x NCCL_TIMEOUT=3600 \
    python3 -u src/train.py \
    --deepspeed examples/deepspeed/ds_z2_config.json \
    --stage sft \
    --model_name_or_path $model_path \
    --do_train \
    --dataset meta2sid_dataset,rec_dataset \
    --use_reasoner \
    --reasoner_layers 1 \
    --template qwen3 \
    --finetuning_type full \
    --output_dir $output_dir \
    --overwrite_cache \
    --save_total_limit 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 200 \
    --learning_rate 1e-4 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --bf16 \
    --dataloader_num_workers 0 \
    --gradient_checkpointing