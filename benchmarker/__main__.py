"""CLI entry point module"""

import sys
import json

# from .benchmarker import run
from benchmarker.util import abstractprocess
from .util.io import save_json


def filter_json_from_output(lines):
    parts = lines.split("benchmarkmagic#!%")
    # for p in parts:
    #     print(p)
    #     print("\n")
    str_json = parts[-1].strip()
    return json.loads(str_json)


def main():
    """CLI entry point function"""
    # run(sys.argv[1:])
    # put corrected args to the command
    # run in a loop to collect counters
    command = ["python3", "-m", "benchmarker.benchmarker"]
    command += sys.argv[1:]
    proc = abstractprocess.Process("local", command=command)
    process_out = proc.get_output()["out"]
    result = filter_json_from_output(process_out)
    print(result)
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
