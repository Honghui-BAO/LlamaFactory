#!/bin/bash

# Configuration (MPI based)
HOSTFILE="/etc/mpi/hostfile"
MASTER_ADDR=$(hostname -I | awk '{print $1}')
MASTER_PORT=12345

# 核心逻辑：解耦 架构路径 和 权重路径
# 1. 架构代码路径（包含你修改过的 modeling_qwen3.py 和 config.json）
model_arch_path="/llm-reco-ssd-share/baohonghui/LlamaFactory/model/qwen3"

# 2. 远程权重路径（只读参数）
remote_weights="/llm-reco-ssd-share/zhangzixing/onerec_pretrain/model_output/pro/sft/8b_v0.1.0_fromstg2_noamazon/step6500/global_step6500/converted"

output_dir="/llm-reco-ssd-share/baohonghui/torchrec/SumRec/merge/ckpt/rmdr_v15_0_single_avg_custom_mpi"
export WANDB_DISABLED=true
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Print Configuration
echo "--------------------------------------"
echo "LlamaFactory MPI + Custom Model Training"
echo "--------------------------------------"
echo "MASTER: $MASTER_ADDR:$MASTER_PORT"
echo "ARCH PATH: $model_arch_path"
echo "WEIGHTS: $remote_weights"
echo "OUTPUT: $output_dir"
echo "--------------------------------------"

# Multi-Node Launch using mpirun
nohup mpirun --allow-run-as-root \
    --hostfile $HOSTFILE \
    -x http_proxy -x https_proxy -x no_proxy \
    -x MASTER_ADDR=$MASTER_ADDR -x MASTER_PORT=$MASTER_PORT \
    -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
    -x NCCL_IB_QPS_PER_CONNECTION=2 \
    -x NCCL_IB_DISABLE=0 \
    -x NCCL_ALGO=^NVLS,NVLSTree \
    -x LD_PRELOAD=/llm-reco-ssd-share/luoxinchen/libs/libnccl.so.2.21.5.noece.cpu \
    -x NCCL_DEBUG=INFO \
    -x NCCL_NVLS_ENABLE=0\
    -x NCCL_SOCKET_IFNAME=eth01 \
    -x NCCL_IB_HCA=mlx5 \
    -x NCCL_PXN_DISABLE=0 \
    -x NCCL_IB_ECE_ENABLE=0\
    -x NCCL_IB_GID_INDEX=3 \
    -x TORCH_NCCL_ENABLE_MONITORING=1 \
    -x TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600 \
    -x TORCH_DISTRIBUTED_DEBUG=DETAIL \
    -x NCCL_ASYNC_ERROR_HANDLING=1 \
    -x TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    -x NCCL_TIMEOUT=3600 \
    python3 -u src/train.py \
    --deepspeed examples/deepspeed/ds_z2_config.json \
    --stage sft \
    --model_name_or_path $model_arch_path \
    --model_weight_path $remote_weights \
    --trust_remote_code True \
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
    --gradient_checkpointing > train_custom_mpi_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo "MPI Training started. Log: train_custom_mpi_*.log"
