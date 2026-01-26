# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from llamafactory.train.tuner import run_exp


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
    
    # Log the detected environment to stdout for debugging
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"--- MPI Env Sync (Rank 0 on Node) ---")
        print(f"RANK: {os.environ.get('RANK')}")
        print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK')}")
        print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
        print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
        print(f"--------------------------------------")


def main():
    sync_mpi_env()
    run_exp()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    sync_mpi_env()
    run_exp()


if __name__ == "__main__":
    main()
