import argparse
import subprocess
import sys
from itertools import product
from pathlib import Path

import yaml


def get_batch_sizes():
    batch_sizes = set()
    for s in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
       batch_sizes.add(s)
    for s in [4 * i for i in range(1, 20)]:
        batch_sizes.add(s)
    for s in [6 * i for i in range(1, 20)]:
        batch_sizes.add(s)
    for s in [32 * i for i in range(1, 21)]:
        batch_sizes.add(s)
    for s in [48 * i for i in range(1, 11)]:
        batch_sizes.add(s)
    return sorted(list(batch_sizes))


def run_all_batches(params):
    defaults = dict()
    defaults["framework"] = "pytorch"
    defaults["nb_epoch"] = "10"
    # TODO: let's focus on GPUS for now
    defaults["gpus"] = "0"
    params.update(defaults)
    for batch_size in get_batch_sizes():
        params["batch_size"] = str(batch_size)
        params["cnt_samples"] = str(batch_size * 10)
        command = ["python3", "-m", "benchmarker", "--preheat"]
        for key, val in params.items():
            command.append(f"--{key}={val}")
        print(command)
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        proc.wait()
        out = proc.stdout.read().decode()
        err = proc.stderr.read().decode()
        if proc.returncode != 0:
            print(out)
            print(err)
        if "out of memory" in err:
            break

def main():
    print("auto run")
    '''
        This in  sense is opposite of benchmarker cli
        there it only runs on particular params you specify
        here it will try to run on everything unless you restrict a specific one(s) to someing
    '''
    path_benchmarks = Path.cwd() / "scripts" / "benchmarks"
    parser = argparse.ArgumentParser(description="Benchmark me up, Scotty!")
    parser.add_argument("--kernel", type=str)
    args = parser.parse_args()
    params_space = []
    precisions = [{"precision": i} for i in ["FP32", "FP16", "TF32"]]
    modes = [{"mode": i} for i in ["inference", "training"]]
    # TODO: why wouldn't kernels themselvs have right defaults for auto benchmark? then we don't need those scripts
    kernels = [{"problem":"roberta_large_mlm", "sample_shape": 256},
               {"problem":"deepcam", "sample:shape": "3,512,512"}]
    params_space.append(precisions)
    params_space.append(modes)
    params_space.append(kernels)
    for i in product(* params_space):
        config = {k: v for d in i for k, v in d.items()}
        print(config)
        run_all_batches(config)

    
if __name__ == "__main__":
    main()