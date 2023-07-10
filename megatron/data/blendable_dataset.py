# Copyright (c) 2021, EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Blendable dataset."""

import time

import numpy as np
import torch

from megatron import print_rank_0
from megatron import mpu


class BlendableDataset(torch.utils.data.Dataset):
    def __init__(self, datasets, weights):
        self.datasets = datasets
        num_datasets = len(datasets)
        assert num_datasets == len(weights)

        self.size = 0       # the total number of samples for training
        for dataset in self.datasets:
            self.size += len(dataset)   # len(dataset) gets the number of samples in this dataset (have been weighted)

        # Normalize weights.
        weights = np.array(weights, dtype=np.float64)
        sum_weights = np.sum(weights)
        assert sum_weights > 0.0
        weights /= sum_weights

        # Build indices.
        start_time = time.time()
        assert num_datasets < 255
        self.dataset_index = np.zeros(self.size, dtype=np.uint8)
        self.dataset_sample_index = np.zeros(self.size, dtype=np.int64)
        import os

        # ld_library_path = os.environ.get('LD_LIBRARY_PATH')
        # if ld_library_path:
        #     print("!!!!!!!!! Pre LD_LIBRARY_PATH:", ld_library_path)
        # else:
        #     print("LD_LIBRARY_PATH is not set.")
            
        # new_ld_library_path = "/home/u2021000178/.conda/envs/llm2/lib:/opt/app/cuda/11.8/lib64:/usr/local/nvidia/lib:"
        # os.environ['LD_LIBRARY_PATH'] = new_ld_library_path
        # updated_ld_library_path = os.environ['LD_LIBRARY_PATH']
        # if ld_library_path:
        #     print("?????????? After LD_LIBRARY_PATH:", updated_ld_library_path)
        # else:
        #     print("LD_LIBRARY_PATH is not set.")
            
        from megatron.data import helpers

        helpers.build_blending_indices(
            self.dataset_index,
            self.dataset_sample_index,
            weights,
            num_datasets,
            self.size,
            torch.distributed.get_rank() == 0,
        )

        print(
            "> RANK {} elapsed time for building blendable dataset indices: "
            "{:.2f} (sec)".format(
                torch.distributed.get_rank(), time.time() - start_time
            )
        )

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        try:
            dataset_idx = self.dataset_index[idx]
            sample_idx = self.dataset_sample_index[idx]
            # print('mkl_pre_idx: {}, {}'.format(idx))
            return self.datasets[dataset_idx][sample_idx]
        except IndexError:
            new_idx = idx % len(self)
            print(
                f"WARNING: Got index out of bounds error with index {idx} - taking modulo of index instead ({new_idx})"
            )
            return self[new_idx]
