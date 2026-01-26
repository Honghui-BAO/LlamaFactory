#!/bin/bash

# Configuration
HOSTFILE="/etc/mpi/hostfile"
MASTER_ADDR=$(hostname -I | awk '{print $1}')
MASTER_PORT=12345

# Paths
model_path=/llm-reco-ssd-share/zhangzixing/onerec_pretrain/model_output/pro/sft/8b_v0.1.0_fromstg2_noamazon/step6500/global_step6500/converted
output_dir=/llm-reco-ssd-share/baohonghui/Reference/torchrec/SumRec/merge/ckpt/rmdr_v15_0_single_avg

# Environment (Based on your RMDR MPI setup)
CONDA_ENV_PATH="/root/miniconda3/envs/rrec"
PYTHON_EXE="$CONDA_ENV_PATH/bin/python3"
export WANDB_DISABLED=true

# Print Configuration
echo "--------------------------------------"
echo "LlamaFactory MPI Training Launch"
echo "--------------------------------------"
echo "MASTER: $MASTER_ADDR:$MASTER_PORT"
echo "OUTPUT: $output_dir"
echo "--------------------------------------"

# Multi-Node Launch using mpirun
# We use the cluster's optimized flags and a bash wrapper for rank mapping
mpirun --allow-run-as-root \
    --hostfile $HOSTFILE \
    -mca opal_set_max_sys_limits 1 \
    -mca btl_openib_allow_ib false \
    -mca plm_rsh_num_concurrent 600 \
    -mca routed_radix 600 \
    -mca pml ob1 \
    -mca btl self,tcp \
    -mca oob_tcp_if_include eth01 \
    -mca btl_tcp_if_include eth01 \
    -x http_proxy -x https_proxy -x no_proxy \
    -x MASTER_ADDR=$MASTER_ADDR -x MASTER_PORT=$MASTER_PORT \
    -x NCCL_DEBUG=WARN \
    -x NCCL_SOCKET_IFNAME=eth01 \
    -x NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8 \
    -x NCCL_IB_GID_INDEX=3 \
    -x OMPI_MCA_btl=self,tcp \
    -x NCCL_NET_PLUGIN=none \
    -x OMPI_MCA_btl_openib_allow_ib=false \
    -x OMPI_MCA_oob_tcp_if_include=eth01 \
    -x OMPI_MCA_pml=ob1 \
    -x NCCL_IB_TIMEOUT=20 \
    -x NCCL_IB_DISABLE=0 \
    -x OMPI_MCA_btl_tcp_if_include=eth01 \
    -x NCCL_NET_OVERHEAD=1000 \
    -x NCCL_IB_QPS_PER_CONNECTION=4 \
    -x PATH="$CONDA_ENV_PATH/bin:$PATH" \
    -x LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:$LD_LIBRARY_PATH" \
    -x PYTHONPATH="$CONDA_ENV_PATH/lib/python3.11/site-packages:$PYTHONPATH" \
    with_nccl_local_env bash -c '
        # Map MPI rank variables to PyTorch variables
        export RANK=${OMPI_COMM_WORLD_RANK:-0}
        export LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK:-0}
        export WORLD_SIZE=${OMPI_COMM_WORLD_SIZE:-1}
        
        # Explicitly set CUDA_VISIBLE_DEVICES to the local rank
        export CUDA_VISIBLE_DEVICES=$LOCAL_RANK
        
        echo "Rank $RANK (Local $LOCAL_RANK) starting on $(hostname)"
        
        $PYTHON_EXE -u src/train.py \
            --deepspeed examples/deepspeed/ds_z2_config.json \
            --stage sft \
            --model_name_or_path '"$model_path"' \
            --do_train \
            --dataset meta2sid_dataset,rec_dataset \
            --use_reasoner \
            --reasoner_layers 1 \
            --template qwen3 \
            --finetuning_type full \
            --output_dir '"$output_dir"' \
            --overwrite_cache \
            --save_total_limit 3 \
            --per_device_train_batch_size 2 \
            --gradient_accumulation_steps 8 \
            --lr_scheduler_type cosine \
            --logging_steps 10 \
            --save_steps 200 \
            --learning_rate 1e-4 \
            --num_train_epochs 3.0 \
            --plot_loss \
            --bf16 \
            --local_rank $LOCAL_RANK
    '
