import json
import os
import datetime


def get_time_str():
    """
    returs current time formatted nicely
    """
    time_now = datetime.datetime.now()
    str_time = time_now.strftime("%y.%m.%d_%H.%M.%S")
    return str_time


def gen_name_output_file(params):
    name = "{}_{}.json".format(
        params["framework"],
        get_time_str())
    return name


def save_json(params):
    str_result = json.dumps(params, sort_keys=True, indent=4, separators=(',', ': '))
    print(str_result)
    path_out = params["path_out"]
    if not os.path.isdir(path_out):
        os.makedirs(path_out)
    name_file = gen_name_output_file(params)
    with open(os.path.join(path_out, name_file), "w") as file_out:
        file_out.write(str_result)
