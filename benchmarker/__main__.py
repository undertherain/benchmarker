"""CLI entry point module"""

import argparse
import json
import os
import sys

from benchmarker import fapp
from benchmarker.nvprof import get_nvprof_counters
from benchmarker.perf import get_counters
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


def main():
    parser = argparse.ArgumentParser(description="Benchmark me up, Scotty!")
    parser.add_argument("--flops", action="store_true")
    parser.add_argument("--fapp_power", action="store_true")
    args, unknown_args = parser.parse_known_args()

    command = ["python3", "-m", "benchmarker.benchmarker"]
    command += unknown_args
    proc = abstractprocess.Process("local", command=command)
    proc_output = proc.get_output()
    returncode = proc_output["returncode"]

    if returncode != 0:
        process_err = proc_output["err"]
        sys.exit(process_err)

    process_out = proc_output["out"]
    result = filter_json_from_output(process_out)
    # TODO: don't parse path_out in the innder loop
    result["platform"] = sysinfo.get_sys_info()
    if result["nb_gpus"] > 0:
        result["device"] = result["platform"]["gpus"][0]["brand"]
        if args.flops:
            result["gflop_measured"] = get_nvprof_counters(command)
    else:
        if (
            "brand" not in result["platform"]["cpu"]
            or result["platform"]["cpu"]["brand"] is None
        ):
            # TODO: add arch when it becomes available thougg sys query
            result["device"] = "unknown CPU"
        else:
            result["device"] = result["platform"]["cpu"]["brand"]
        if args.flops:
            result["gflop_measured"] = get_counters(command)
        elif args.fapp_power:
            total, details = fapp.get_power_total_and_detail(command)
            result["problem"]["power"] = {"total": total, "details": details}
    cute_device = get_cute_device_str(result["device"]).replace(" ", "_")
    result["path_out"] = os.path.join(result["path_out"], result["problem"]["name"])
    result["path_out"] = os.path.join(result["path_out"], cute_device)
    if "gflop_measured" in result.keys():
        result["gflop_per_joule"] = (
            result["gflop_measured"] / result["power"]["avg_watt_total"]
        )
    save_json(result)
    # TODO: don't measure power when measureing flops


if __name__ == "__main__":
    main()
