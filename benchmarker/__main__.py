"""CLI entry point module"""

import sys
import json
import argparse
import os
from .util import sysinfo

# from .benchmarker import run
from benchmarker.util import abstractprocess
from benchmarker.util.cute_device import get_cute_device_str
from .util.io import save_json


def filter_json_from_output(lines):
    parts = lines.split("benchmarkmagic#!%")
    # for p in parts:
    #     print(p)
    #     print("\n")
    str_json = parts[-1].strip()
    return json.loads(str_json)


def main():
    parser = argparse.ArgumentParser(description='Benchmark me up, Scotty!')
    parser.add_argument('--path_out', type=str, default="./logs")
    parser.add_argument("--problem")
    args, unknown_args = parser.parse_known_args()

    """CLI entry point function"""
    # run(sys.argv[1:])
    # put corrected args to the command
    # run in a loop to collect counters
    command = ["python3", "-m", "benchmarker.benchmarker"]
    command += sys.argv[1:]
    proc = abstractprocess.Process("local", command=command)
    process_out = proc.get_output()["out"]
    result = filter_json_from_output(process_out)
    # TODO: don't parse path_out in the innder loop
    result["platform"] = sysinfo.get_sys_info()
    if result["nb_gpus"] > 0:
        result["device"] = result["platform"]["gpus"][0]["brand"]
    else:
        if result["platform"]["cpu"]["brand"] is not None:
            result["device"] = result["platform"]["cpu"]["brand"]
        else:
            # TODO: add arch when it becomes available thougg sys query
            result["device"] = "unknown CPU"
            result["path_out"] = args.path_out
    cute_device = get_cute_device_str(result["device"]).replace(" ", "_")
    result["path_out"] = os.path.join(result["path_out"], result["problem"]["name"])
    result["path_out"] = os.path.join(result["path_out"], cute_device)
    save_json(result)
    # TODO: move system_query here

    # add to results
    # save
    #Run benchmarker with params and get output as dict
    # TODO: don't measure power when we are collecting perf
    #output_string, error_string = run(params, command)
    #output_dict = json.loads(output_string)
    #print(output_dict)

    #Collect counters, add to the dict, save as json
    #dict_with_ops = get_counters(params, output_dict, command)
    #cute_device = get_cute_device_str(dict_with_ops["device"]).replace(" ", "_")
    #dict_with_ops["path_out"] = os.path.join("./flops",dict_with_ops["problem"]["name"],dict_with_ops["mode"], cute_device)
    #print(dict_with_ops)
    #save_json(dict_with_ops)


if __name__ == "__main__":
    main()
