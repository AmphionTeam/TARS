#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import sys

from datasets import load_dataset


# Use resolve/main instead of blob/main
FILE_URL = (
    "https://huggingface.co/datasets/yuantuo666/"
    "MMLU_FULL-full_train_dpo_hf_format.catm2ram2.v1/"
    "resolve/main/data/train-00003-of-00104.parquet"
)
FILE_NAME = "train-00003-of-00104.parquet"

TEST_DIR = "test_dataset"
TRAIN_DIR = "train_dataset"


def download_file(url: str, output_path: str):
    """Download the file using wget."""
    if os.path.exists(output_path):
        print(f"[Info] File already exists, skipping download: {output_path}")
        return

    cmd = ["wget", "-O", output_path, url]
    print("[Info] Starting file download...")
    print(" ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
        print(f"[Info] Download completed: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"[Error] wget download failed: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("[Error] wget was not found. Please install wget first.")
        sys.exit(1)


def main():
    # 1. Download the parquet file
    download_file(FILE_URL, FILE_NAME)

    # 2. Load the parquet dataset
    print("[Info] Loading parquet dataset...")
    ds_dict = load_dataset("parquet", data_files=FILE_NAME)
    ds = ds_dict["train"]

    print("[Info] Full dataset:")
    print(ds)
    print()

    # 3. Split the dataset
    test = ds.select(range(0, 10))
    train = ds.select(range(10, len(ds)))

    print("[Info] Test dataset:")
    print(test)
    print()

    print("[Info] Train dataset:")
    print(train)
    print()

    # 4. Save to disk
    print(f"[Info] Saving test dataset to {TEST_DIR} ...")
    test.save_to_disk(TEST_DIR)

    print(f"[Info] Saving train dataset to {TRAIN_DIR} ...")
    train.save_to_disk(TRAIN_DIR)

    print("[Info] Done.")
    print(f"Run: export DATASET_PATH={TEST_DIR}")
    print(f"Run: export DATASET_PATH={TRAIN_DIR}")


if __name__ == "__main__":
    main()
