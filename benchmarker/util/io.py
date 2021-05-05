import datetime
import json
import os
import re


def get_cute_device_str(device_name):
    shorts = [
        "(GTX 980 Ti)",
        "(GTX \d\d\d\d Ti)",
        "(RTX \d\d\d\d Ti)",
        "(GTX \d\d\d\d)(?: \dGB)",
        "(RTX \d\d\d\d)",
        "(P100-PCIE)",
        "(P100-SXM2)",
        "(V100-SXM2)",
        "(K20Xm)",
        "(K40c)",
        "(ThunderX2)",
        "(Xeon)(?:\(R\) CPU )(E5-[0-9]{4} v[0-9])",
        "(Xeon)(?:\(R\)) (Gold) ([0-9]{4})",  # 'Intel(R)_Xeon(R)_Gold_6148_CPU_@_2.40GHz'
        "(Core)(?:\(TM\) )(i5-[0-9]{4}[A-Z])",
        "(?:AMD) (Ryzen [0-9] [0-9]{4}[A-Z])",
        "(?:AMD) (Ryzen [0-9] [0-9]{4})",
        "(?:AMD) (EPYC [0-9]{4})",
        "i7-3820",
        "i7-3930K",
    ]
    for short in shorts:
        m = re.search(short, device_name)
        if m:
            found = " ".join(m.groups())
            return found
    return device_name


def get_time_str():
    """
    returs current time formatted nicely
    """
    time_now = datetime.datetime.now()
    str_time = time_now.strftime("%y.%m.%d_%H.%M.%S")
    return str_time


def gen_name_output_file(params):
    return "{}_{}.json".format(params["framework"], params["start_time"])


def get_path_out_dir(params):
    path_out = os.path.join(
        params["path_out"],
        params["problem"]["name"],
        get_cute_device_str(params["device"]).replace(" ", "_"),
    )
    if not os.path.isdir(path_out):
        os.makedirs(path_out)
    return path_out


def save_json(params):
    str_result = json.dumps(params, sort_keys=True, indent=4, separators=(",", ": "))
    print(str_result)
    path_out = get_path_out_dir(params)
    name_file = gen_name_output_file(params)
    with open(os.path.join(path_out, name_file), "w") as file_out:
        file_out.write(str_result)


def print_json(params):
    str_result = json.dumps(params, sort_keys=True, indent=4, separators=(",", ": "))
    print(str_result)
