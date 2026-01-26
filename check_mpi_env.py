import os
import torch
import socket

def print_env():
    rank = os.environ.get('RANK', 'N/A')
    local_rank = os.environ.get('LOCAL_RANK', 'N/A')
    world_size = os.environ.get('WORLD_SIZE', 'N/A')
    ompi_rank = os.environ.get('OMPI_COMM_WORLD_RANK', 'N/A')
    pmi_rank = os.environ.get('PMI_RANK', 'N/A')
    
    hostname = socket.gethostname()
    gpu_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device() if gpu_count > 0 else 'N/A'
    
    print(f"[Host: {hostname}] RANK: {rank}, LOCAL_RANK: {local_rank}, WORLD_SIZE: {world_size}, OMPI_RANK: {ompi_rank}, PMI_RANK: {pmi_rank}, GPUs: {gpu_count}, Current: {current_device}")

if __name__ == "__main__":
    print_env()
