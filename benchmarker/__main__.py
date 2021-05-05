"""CLI entry point module"""

import argparse
import os
import sys

import benchmarker.benchmarker
from benchmarker.profiling import fapp, nvprof, perf
from benchmarker.results import add_result_details

from .util import sysinfo
from .util.io import get_cute_device_str, get_time_str, save_json


def get_args():
    parser = argparse.ArgumentParser(description="Benchmark me up, Scotty!")
    parser.add_argument("--flops", action="store_true", default=False)
    parser.add_argument("--power_fapp", action="store_true", default=False)
    # removed: see issue #167
    # parser.add_argument("--profile_pytorch", action="store_true")
    return parser.parse_known_args()


# removed: see issue #167
# def filter_json_from_output(lines):
#     parts = lines.split("benchmarkmagic#!%")
#     # for p in parts:
#     #     print(p)
#     #     print("\n")
#     str_json = parts[-1].strip()
#     return json.loads(str_json)


# def run_cmd_and_get_output(command):
#     proc = abstractprocess.Process("local", command=command)
#     proc_output = proc.get_output()
#     returncode = proc_output["returncode"]

#     if returncode != 0:
#         process_err = proc_output["err"]
#         sys.exit(process_err)

#     process_out = proc_output["out"]
#     result = filter_json_from_output(process_out)
#     return result


def main():
    args, unknown_args = get_args()
    command = [sys.executable, "-m", "benchmarker"]
    command += unknown_args
    # TODO(vatai): remove rapl and nvml and any other profiling. Make
    # a base class (or something) which ignores this)
    result = benchmarker.benchmarker.run(unknown_args)

    # TODO: don't parse path_out in the innder loop
    result["platform"] = sysinfo.get_sys_info()
    result["start_time"] = get_time_str()
    if result["nb_gpus"] > 0:
        precision = result["problem"]["precision"]
        result["device"] = result["platform"]["gpus"][0]["brand"]
        if args.flops and precision in ["FP16", "FP32"]:
            result["problem"]["gflop_measured"] = nvprof.get_gflop(command, precision)
    else:
        if (
            "brand" not in result["platform"]["cpu"]
            or result["platform"]["cpu"]["brand"] is None
        ):
            # TODO: add arch when it becomes available thougg sys query
            result["device"] = "unknown CPU"
        else:
            result["device"] = result["platform"]["cpu"]["brand"]
        if args.flops and "Intel" in result["device"]:
            result["problem"]["gflop_measured"] = perf.get_gflop(command)
        elif args.power_fapp:
            avg_watt_total, details = fapp.get_power_total_and_detail(command)
            result["power"] = {"avg_watt_total": avg_watt_total, "details": details}

    # removed: see issue #167
    # Collect profile data when profile_pytorch switch is enabled
    # if args.profile_pytorch:
    #     command += ["--profile_pytorch"]
    #     profile_result = run_cmd_and_get_output(command)
    #     result["profile_pytorch"] = True
    #     result["profile_data"] = profile_result["profile_data"]
    #     result["path_out"] = os.path.join(result["path_out"], "profile")

    cute_device = get_cute_device_str(result["device"]).replace(" ", "_")
    result["path_out"] = os.path.join(result["path_out"], result["problem"]["name"])
    result["path_out"] = os.path.join(result["path_out"], cute_device)

    # TODO: call fill_result_details here
    add_result_details(result)
    print(result)
    save_json(result)
    # TODO: don't measure power when measureing flops


if __name__ == "__main__":
    main()
