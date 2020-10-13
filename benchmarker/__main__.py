"""CLI entry point module"""

import sys

from .benchmarker import run
# from benchmarker.util import abstractprocess
from .util.io import save_json


def main():
    """CLI entry point function"""
    # run(sys.argv[1:])
    # put corrected args to the command
    # run in a loop to collect counters
    #command = "run benchmarker"
    #proc = abstractprocess.Process("local", command=command)
    #result = proc.get_output()
    # add to results
    # save
    #Run benchmarker with params and get output as dict
    output_string, error_string = run(params, command)
    output_dict = json.loads(output_string)
    #print(output_dict)

    #Collect counters, add to the dict, save as json
    dict_with_ops = get_counters(params, output_dict, command)
    cute_device = get_cute_device_str(dict_with_ops["device"]).replace(" ", "_")
    dict_with_ops["path_out"] = os.path.join("./flops",dict_with_ops["problem"]["name"],dict_with_ops["mode"], cute_device)
    #print(dict_with_ops)
    save_json(dict_with_ops)


if __name__ == "__main__":
    main()
