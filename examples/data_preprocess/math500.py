# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Preprocess the MATH-500 dataset to parquet format
"""

import argparse
import json
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/math")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    # 'lighteval/MATH' is no longer available on huggingface.
    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    data_source = "HuggingFaceH4/MATH-500"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    # MATH-500 数据集，检查可用的分割
    print(f"Available splits: {list(dataset.keys())}")
    
    # 如果只有一个分割，直接使用；否则合并所有分割
    if len(dataset.keys()) == 1:
        split_name = list(dataset.keys())[0]
        all_dataset = dataset[split_name]
        print(f"Using single split: {split_name}")
    else:
        # 合并所有分割
        from datasets import concatenate_datasets
        all_dataset = concatenate_datasets([dataset[split] for split in dataset.keys()])
        print(f"Concatenated all splits: {list(dataset.keys())}")

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn():
        def process_fn(example, idx):
            question = example.pop("problem")

            question = question + " " + instruction_following

            answer = example.pop("solution")
            solution = extract_solution(answer)
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {"index": idx},
            }
            return data

        return process_fn

    all_dataset = all_dataset.map(function=make_map_fn(), with_indices=True)

    local_dir = os.path.expanduser(args.local_dir)
    hdfs_dir = args.hdfs_dir

    # 保存所有数据
    all_dataset.to_parquet(os.path.join(local_dir, "math500_test.parquet"))
    # Save one example as JSON for reference
    example = all_dataset[0]
    with open(os.path.join(local_dir, "math500_example.json"), "w") as f:
        json.dump(example, f, indent=2)
    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
