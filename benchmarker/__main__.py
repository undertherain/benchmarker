"""CLI entry point module"""

import argparse
import json
import os
import sys

from benchmarker import fapp, nvprof, perf
# from .benchmarker import run
from benchmarker.util import abstractprocess
from benchmarker.util.cute_device import get_cute_device_str

from .util import sysinfo
from .util.io import save_json


def filter_json_from_output(lines):
    parts = lines.split("benchmarkmagic#!%")
    # for p in parts:
    #     print(p)
    #     print("\n")
    str_json = parts[-1].strip()
    return json.loads(str_json)

def run_cmd_and_get_output(command):
    proc = abstractprocess.Process("local", command=command)
    proc_output = proc.get_output()
    returncode = proc_output["returncode"]

    if returncode != 0:
        process_err = proc_output["err"]
        sys.exit(process_err)

    process_out = proc_output["out"]
    result = filter_json_from_output(process_out)
    return result

def main():
    parser = argparse.ArgumentParser(description="Benchmark me up, Scotty!")
    parser.add_argument("--flops", action="store_true")
    parser.add_argument("--fapp_power", action="store_true")
    parser.add_argument("--profile_pytorch", action="store_true")
    args, unknown_args = parser.parse_known_args()
    command = ["python3", "-m", "benchmarker.benchmarker"]
    command += unknown_args
    result = run_cmd_and_get_output(command)
    # TODO: don't parse path_out in the innder loop
    result["platform"] = sysinfo.get_sys_info()
    if result["nb_gpus"] > 0:
        result["device"] = result["platform"]["gpus"][0]["brand"]
        if args.flops and result["problem"]["precision"] in ["FP16", "FP32"]:
            result["problem"]["gflop_measured"] = nvprof.get_gflop(command, result["problem"]["precision"])
    else:
        if (
            "brand" not in result["platform"]["cpu"]
            or result["platform"]["cpu"]["brand"] is None
        ):
            # TODO: add arch when it becomes available thougg sys query
            result["device"] = "unknown CPU"
        else:
            result["device"] = result["platform"]["cpu"]["brand"]
        if args.flops and 'Intel' in result["device"]:
            result["problem"]["gflop_measured"] = perf.get_gflop(command)
        elif args.fapp_power:
            avg_watt_total, details = fapp.get_power_total_and_detail(command)
            result["power"] = {"avg_watt_total": avg_watt_total, "details": details}
    # Collect profile data when profile_pytorch switch is enabled
    if args.profile_pytorch:
        command += ["--profile_pytorch"]
        profile_result = run_cmd_and_get_output(command)
        result["profile_pytorch"] = True
        result["profile_data"] = profile_result["profile_data"]
        result["path_out"] = "./logs/profile"
         
    cute_device = get_cute_device_str(result["device"]).replace(" ", "_")
    result["path_out"] = os.path.join(result["path_out"], result["problem"]["name"])
    result["path_out"] = os.path.join(result["path_out"], cute_device)
    if "gflop_measured" in result["problem"]:
        if result["power"]["joules_total"] != 0:
            result["gflop_per_joule"] = float(result["problem"]["gflop_measured"])
            result["gflop_per_joule"] /= float(result["power"]["joules_total"])
        if "gflop_per_second" not in result:
            result["gflop_per_second"] = float(result["problem"]["gflop_measured"]) / result["time_total"]
        else:
            result["gflop_per_second_measured"] = float(result["problem"]["gflop_measured"]) / result["time_total"]
    save_json(result)
    # TODO: don't measure power when measureing flops


if __name__ == "__main__":
    main()
