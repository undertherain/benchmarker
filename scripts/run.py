#!/usr/bin/env python3
import multiprocessing
import subprocess

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
    params["problem_size"] = params["batch_size"] * 4
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
