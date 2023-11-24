import subprocess
from pathlib import Path

import yaml


def run_single_instance(params):
    command = ["python3", "-m", "benchmarker", "--preheat", "--framework=pytorch"]
    params["batch_size"] = 3
    params["cnt_samples"] = params["batch_size"] * 10
    for key, val in params.items():
        command.append(f"--{key}={val}")
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc.wait()
    out = proc.stdout.read().decode()
    err = proc.stderr.read().decode()
    if proc.returncode != 0:
        print(out)
        print(err)

def run_benchmark(config):
    print(config)
    run_single_instance(config)
            # iterate all precisions
    #--power_nvml
    #     --batch_size=64
    #--nb_epoch=10
    #--preheat
    #--mode=training
    #--precision=TF32
    #--gpus=0


def main():
    print("auto run")
    path_benchmarks = Path.cwd() / "scripts" / "benchmarks"
    for path_config_benchmark in path_benchmarks.iterdir():
        with open(path_config_benchmark) as f:
            config_benchmark = yaml.load(f)
            run_benchmark(config_benchmark)
     # READ general config girst, i.e. which devices to use?

    
if __name__ == "__main__":
    main()