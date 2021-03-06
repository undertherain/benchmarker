#!/usr/bin/env python3

"""This file can be used to run benchmarker with multiple batch sizes
by specifying --batch_size=fast or --batch_size=slow.  In this case
the --problem_size is automatically generated (see
`prob_size_multiplier`) and the rest of the arguments (--problem,
--framework etc) are passed to benchmarker.

"""

import argparse
import multiprocessing
import subprocess

prob_size_multiplier = 4

fast_batches = set()
fast_batches.update(range(1, 10))
fast_batches.add(multiprocessing.cpu_count())
fast_batches.add(multiprocessing.cpu_count() // 2)
fast_batches.update([16, 32, 64, 128, 256])
fast_batches = sorted(list(fast_batches))

slow_batches = list(fast_batches)
slow_batches += list(range(10, 32, 2))
slow_batches += list(range(32, 64, 4))
slow_batches += list(range(64, 128, 8))
slow_batches += list(range(128, 256, 16))
slow_batches = sorted(list(set(slow_batches)))


def run(params, argv=["benchmarker"]):
    command = ["python3", "-m"] + argv
    for key, val in params.items():
        command.append(f"--{key}={val}")
    # print(command)
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc.wait()
    out = proc.stdout.read().decode()
    err = proc.stderr.read().decode()
    if proc.returncode != 0:
        print(out)
        print(err)


def run_on_all_backends(params):
    params["problem_size"] = params["batch_size"] * prob_size_multiplier
    params["framework"] = "tensorflow"
    run(params)
    params["gpus"] = "0"
    run(params)
    params["framework"] = "pytorch"
    params["backend"] = "native"
    run(params)
    params.pop("gpus")
    run(params)
    params["backend"] = "DNNL"
    run(params)


def run_batch_size(batch_size, argv):
    params = {
        "batch_size": batch_size,
        "problem_size": batch_size * prob_size_multiplier,
    }
    run(params, ["benchmarker"] + argv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size")
    args, unknown_args = parser.parse_known_args()

    assert args.batch_size is not None, "--batch_size required"

    if args.batch_size.isnumeric():
        run_batch_size(args.batch_size, unknown_args)
    else:
        has_prob_size = any(map(lambda t: t.startswith("--problem_size"), unknown_args))
        assert not has_prob_size, "--problem_size should NOT be specified"
        for batch_size in eval("{}_batches".format(args.batch_size)):
            run_batch_size(batch_size, unknown_args)
