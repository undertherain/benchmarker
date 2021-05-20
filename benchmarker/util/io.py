import datetime
import json
import os

from .cute_device import get_cute_device_str


def get_time_str():
    """
    returs current time formatted nicely
    """
    time_now = datetime.datetime.now()
    str_time = time_now.strftime("%y.%m.%d_%H.%M.%S")
    return str_time


def get_path_out_name(params, ext="json"):
    return f'{params["framework"]}_{params["start_time"]}.{ext}'


def get_path_out_dir(params):
    path_out = os.path.join(
        params["path_out"],
        params["path_ext"],
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
    name_file = get_path_out_name(params)
    with open(os.path.join(path_out, name_file), "w") as file_out:
        file_out.write(str_result)


def print_json(params):
    str_result = json.dumps(params, sort_keys=True, indent=4, separators=(",", ": "))
    print(str_result)
