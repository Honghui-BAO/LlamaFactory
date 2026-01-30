#!/bin/bash

# Configuration
HOSTFILE="/etc/mpi/hostfile"
MASTER_ADDR=$(hostname -I | awk '{print $1}')
MASTER_PORT=12345

# Paths
export MODEL_PATH="/llm-reco-ssd-share/baohonghui/ckpt/converted"
export DATA_PATH="/llm-reco-ssd-share/baohonghui/LlamaFactory/data/sum_rec/amazon_merged_sft_data_rec_honghui_v2.json"
export OUTPUT_DIR="/llm-reco-ssd-share/baohonghui/torchrec/SumRec/merge/ckpt/reasoner_grpo_v1"
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

echo "--------------------------------------"
echo "LlamaFactory GRPO MPI Training Launch"
echo "--------------------------------------"
echo "MASTER: $MASTER_ADDR:$MASTER_PORT"
echo "OUTPUT: $OUTPUT_DIR"
echo "--------------------------------------"

# Multi-Node Launch using mpirun
mpirun --allow-run-as-root \
    --hostfile $HOSTFILE \
    -x http_proxy -x https_proxy -x no_proxy \
    -x MASTER_ADDR=$MASTER_ADDR -x MASTER_PORT=$MASTER_PORT \
    -x LD_LIBRARY_PATH=$LIBRARY_PATH \
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
    -x MODEL_PATH=$MODEL_PATH -x DATA_PATH=$DATA_PATH -x OUTPUT_DIR=$OUTPUT_DIR \
    python3 -u honghui_grpo_train.py 
