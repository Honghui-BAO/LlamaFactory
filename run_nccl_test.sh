#!/bin/bash

# Configuration (Based on your honghui_train_8b_mpi.sh)
HOSTFILE="/etc/mpi/hostfile"
MASTER_ADDR=$(hostname -I | awk '{print $1}')
MASTER_PORT=12345

# Environment
CONDA_ENV_PATH="/root/miniconda3/envs/rrec"
PYTHON_EXE="$CONDA_ENV_PATH/bin/python3"

echo "--------------------------------------"
echo "NCCL Multi-Node Connectivity Test"
echo "MASTER: $MASTER_ADDR:$MASTER_PORT"
echo "HOSTFILE: $HOSTFILE"
echo "--------------------------------------"

# Launch Test using mpirun
mpirun --allow-run-as-root \
    --hostfile $HOSTFILE \
    -x MASTER_ADDR=$MASTER_ADDR -x MASTER_PORT=$MASTER_PORT \
    -x NCCL_DEBUG=INFO \
    -x NCCL_SOCKET_IFNAME=eth01 \
    -x NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8 \
    -x NCCL_IB_GID_INDEX=3 \
    -x PATH="$CONDA_ENV_PATH/bin:$PATH" \
    -x LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:$LD_LIBRARY_PATH" \
    -x PYTHONPATH="$CONDA_ENV_PATH/lib/python3.11/site-packages:." \
    $PYTHON_EXE test_nccl.py
