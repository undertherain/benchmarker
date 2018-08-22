# -*- coding: utf-8 -*-
"""Benchmarker main module

This is where all magic is happening
"""

import importlib
import json
import os
import datetime
import pkgutil
# import logging

from .util import sysinfo


def get_time_str():
    """
    returs current time formatted nicely
    """
    time_now = datetime.datetime.now()
    str_time = time_now.strftime("%y.%m.%d_%H.%M.%S")
    return str_time


def gen_name_output_file(params):
    name = "{}_{}_{}_{}.json".format(
        params["problem"]["name"],
        params["framework"],
        params["device"],
        get_time_str()
        )
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


def get_modules():
    path_modules = "benchmarker/modules"
    return [name for _, name, is_pkg in pkgutil.iter_modules([path_modules])
            if not is_pkg and name.startswith('do_')]


def run(args):
    params = {}
    params["platform"] = sysinfo.get_sys_info()
    params["path_out"] = args.path_out

    if args.framework is None:
        print("please choose one of the frameworks to evaluate")
        print("available frameworks:")
        for plugin in get_modules():
            print("\t", plugin[3:])
        return

    # todo: load frameowrk's metadata from backend
    params["framework"] = args.framework

    if args.problem is None:
        print("choose a problem to run")
        print("problems supported by {}:".format(args.framework))
        return
    # todo: get a list of support problems for a given framework

    # todo: load problem's metadata from the problem itself
    params["problem"] = {}
    params["problem"]["name"] = args.problem
    if args.size is not None:
        params["problem"]["size"] = int(args.size)
    params["misc"] = args.misc
    if args.gpus:
        params["gpus"] = list(map(int, args.gpus.split(',')))
    else:
        params["gpus"] = []

    params["nb_gpus"] = len(params["gpus"])

    if params["nb_gpus"] > 0:
        params["device"] = params["platform"]["gpus"][0]["brand"]
    else:
        params["device"] = params["platform"]["cpu"]["brand"]

    mod = importlib.import_module("benchmarker.modules.do_" + params["framework"])
    run = getattr(mod, 'run')

    params = run(params)
    save_json(params)
