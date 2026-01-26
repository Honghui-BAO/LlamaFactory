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
    # Mapping MPI env vars to standard ones
    mapping = {
        "OMPI_COMM_WORLD_RANK": "RANK",
        "OMPI_COMM_WORLD_LOCAL_RANK": "LOCAL_RANK",
        "OMPI_COMM_WORLD_SIZE": "WORLD_SIZE",
        "PMI_RANK": "RANK",
        "PMI_LOCALRANKID": "LOCAL_RANK",
        "PMI_SIZE": "WORLD_SIZE",
    }
    for mpi_var, std_var in mapping.items():
        if mpi_var in os.environ and std_var not in os.environ:
            os.environ[std_var] = os.environ[mpi_var]


def main():
    sync_mpi_env()
    run_exp()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    sync_mpi_env()
    run_exp()


if __name__ == "__main__":
    main()
