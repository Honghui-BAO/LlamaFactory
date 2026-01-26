import os
import torch
import socket
import sys

def print_env():
    # Capture all env vars that might be related to MPI or Rank
    keys = [
        'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 
        'OMPI_COMM_WORLD_RANK', 'OMPI_COMM_WORLD_LOCAL_RANK', 'OMPI_COMM_WORLD_SIZE',
        'PMI_RANK', 'PMI_LOCALRANKID', 'PMI_SIZE',
        'SLURM_PROCID', 'SLURM_LOCALID', 'SLURM_NTASKS',
        'MV2_COMM_WORLD_RANK', 'MV2_COMM_WORLD_LOCAL_RANK'
    ]
    
    env_summary = {k: os.environ.get(k, 'N/A') for k in keys}
    
    hostname = socket.gethostname()
    gpu_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device() if gpu_count > 0 else 'N/A'
    
    print(f"--- Process Info ---")
    print(f"Hostname: {hostname}")
    print(f"GPUs available: {gpu_count}")
    print(f"Current Device: {current_device}")
    for k, v in env_summary.items():
        if v != 'N/A':
            print(f"{k}: {v}")
    
    # If rank is available, print only once for rank 0 or just print for all if count is small
    print(f"--------------------\n")

if __name__ == "__main__":
    print_env()
