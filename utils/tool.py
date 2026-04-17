import os
import json
import time
import hashlib
import sys


def test_until_exist(path):
    while True:
        time.sleep(0.1)
        if os.path.exists(path):
            time.sleep(1)
            return True


def test_until_not_exist(path):
    while True:
        time.sleep(0.1)
        if not os.path.exists(path):
            time.sleep(1)
            return True


def get_program_hash():
    info = "command=" + " ".join(sys.argv) + "&cwd=" + os.getcwd()
    return hashlib.md5(info.encode("utf-8")).hexdigest()


global_rank = None
global_world_size = None


def test_gather_list(rank, world_size):
    print(f"INFO: test_gather_list testing, rank={rank}, world_size={world_size}")
    r = gather_list([rank], rank, world_size)
    if rank == 0:
        assert r == list(range(world_size)), f"Problemetic in gather_list, r={r}"
    print(f"INFO: test_gather_list OK, rank={rank}, world_size={world_size}")
    return True


def gather_list(list, rank=None, world_size=None):
    global global_rank, global_world_size

    if not rank and global_rank:
        rank = global_rank
    if not world_size and global_world_size:
        world_size = global_world_size

    hashcode = get_program_hash()
    save_path = f"/tmp/gather_list_{hashcode}_{rank}_of_{world_size}.json"

    if os.path.exists(save_path):
        print(f"WANGING: exist file found, deleting: {save_path}")
        os.remove(save_path)

    with open(save_path, "w") as f:
        json.dump(list, f)

    if rank != 0:
        test_until_not_exist(save_path)
        return save_path

    # gather all file
    results = []
    for i in range(world_size):
        read_path = f"/tmp/gather_list_{hashcode}_{i}_of_{world_size}.json"
        test_until_exist(read_path)
        success = False
        for i in range(3):
            try:
                with open(read_path, "r") as f:
                    data = json.load(f)
                    results.extend(data)
                    success = True
                    break
            except:
                time.sleep(10)
                print(f"Failed to open {read_path}, wating 10s and retry. i={i}")
                pass
        if success:
            os.remove(read_path)

    return results


def print_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        "=" * 80,
        f"\nTrainable params: {trainable_params:,} || Total params: {total_params:,} || Trainable percentage: {100 * trainable_params / total_params:.4f}%\n",
        "=" * 80,
    )
