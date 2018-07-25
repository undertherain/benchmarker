import importlib
import json
import os
import datetime
import sys
import pkgutil
import logging

from .util import sysinfo


def get_time_str():
    d = datetime.datetime.now()
    s = d.strftime("%y.%m.%d_%H.%M.%S")
    return s


def gen_name_output_file(params):
    name = "{}_{}_{}_{}.json".format(
        params["problem"],
        params["framework"],
        params["device"],
        get_time_str()
        )
    return name


def save_json(params):
    str_result = json.dumps(params, sort_keys=True,  indent=4, separators=(',', ': '))
    print(str_result)
    path_out = params["path_out"]
    if not os.path.isdir(path_out):
        os.makedirs(path_out)
    name_file = gen_name_output_file(params)
    with open(os.path.join(path_out, name_file), "w") as f:
        f.write(str_result)


def get_modules():
    path_modules = "benchmarker/modules"
    return [name for _, name, is_pkg in pkgutil.iter_modules([path_modules]) if not is_pkg and name.startswith('do_')]


def run(framework: "Framework to test" = None,
         problem: "problem to solve" = None,
         path_out: "path to store results" = "./logs",
         gpus: "list of gpus to use" = "",
         misc: "comma separated list of key:value pairs" = None
         ):

    if framework is None:
        print("please choose one of the frameworks to evaluate")
        print("available frameworks:")
        for plugin in get_modules():
            print("\t", plugin[3:])
        return

    if problem is None:
        print("choose a problem to run")
        print(f"problems supported by {framework}:")
        return
    # get list of support problems for a given framework

    params = {}
    params["platform"] = sysinfo.get_sys_info()
    params["framework"] = framework
    params["path_out"] = path_out
    params["problem"] = problem
    params["misc"] = misc
    if len(gpus) > 0:
        params["gpus"] = list(map(int, gpus.split(',')))
    else:
        params["gpus"] = []

    params["nb_gpus"] = len(params["gpus"])

    if params["nb_gpus"] > 0:
        params["device"] = params["platform"]["gpus"][0]["brand"]
    else:
        params["device"] = params["platform"]["cpu"]["brand"]

    mod = importlib.import_module("benchmarker.modules.do_"+params["framework"])
    run = getattr(mod, 'run')

    params = run(params)
    save_json(params)
