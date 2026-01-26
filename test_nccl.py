import os
import torch
import torch.distributed as dist
import socket
import datetime

def sync_mpi_env():
    # Comprehensive Mapping for multi-vendor MPI and Schedulers
    mapping = {
        "RANK": ["OMPI_COMM_WORLD_RANK", "PMI_RANK", "SLURM_PROCID", "MV2_COMM_WORLD_RANK"],
        "LOCAL_RANK": ["OMPI_COMM_WORLD_LOCAL_RANK", "PMI_LOCALRANKID", "SLURM_LOCALID", "MV2_COMM_WORLD_LOCAL_RANK"],
        "WORLD_SIZE": ["OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS", "MV2_COMM_WORLD_SIZE"],
    }
    for std_var, mpi_vars in mapping.items():
        if std_var not in os.environ:
            for mpi_var in mpi_vars:
                if mpi_var in os.environ:
                    os.environ[std_var] = os.environ[mpi_var]
                    break

def test_connectivity():
    sync_mpi_env()
    
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    hostname = socket.gethostname()
    
    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        print(f"Rank {rank} (Local {local_rank}) on {hostname}: CUDA NOT AVAILABLE")
        return

    print(f"Rank {rank} (Local {local_rank}) on {hostname}: Initializing NCCL (World Size: {world_size})...")
    
    try:
        # Initialize process group
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(seconds=600)
        )
        
        # Simple All-Reduce test
        tensor = torch.ones(1).to(device) * (rank + 1)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        
        # Wait for all to finish
        dist.barrier()
        
        expected_sum = sum(range(1, world_size + 1))
        if rank == 0:
            print(f"\n[SUCCESS] NCCL Test Completed!")
            print(f"Result Sum: {tensor.item()} (Expected: {expected_sum})")
            print(f"Total ranks participated: {world_size}")
            
    except Exception as e:
        print(f"Rank {rank} on {hostname}: FAILED with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    test_connectivity()
