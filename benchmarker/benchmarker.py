# -*- coding: utf-8 -*-
"""Benchmarker main module

This is where all magic is happening
"""

import importlib
import os
import pkgutil
# import logging

from .util import sysinfo
from .util.io import save_json


def get_modules():
    path_modules = "benchmarker/modules"
    return [name for _, name, is_pkg in pkgutil.iter_modules([path_modules])
            if not is_pkg and name.startswith('do_')]


def run(args, unknown_args):
    params = {}
    params["platform"] = sysinfo.get_sys_info()

    if args.framework is None:
        print("please choose one of the frameworks to evaluate")
        print("available frameworks:")
        for plugin in get_modules():
            print("\t", plugin[3:])
        return

    # TODO: load frameowork's metadata from backend
    params["framework"] = args.framework
    params["path_out"] = args.path_out

    if args.problem is None:
        print("choose a problem to run")
        print("problems supported by {}:".format(args.framework))
        return
    # TODO: get a list of support problems for a given framework

    # TODO: load problem's metadata from the problem itself
    params["problem"] = {}
    params["problem"]["name"] = args.problem
    if args.problem_size is not None:
        params["problem"]["size"] = int(args.problem_size)
    if args.batch_size is not None:
        params["batch_size_per_device"] = int(args.batch_size)
    # params["misc"] = args.misc
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
    benchmark = getattr(mod, 'Benchmark')(params, unknown_args)
    params = benchmark.run()
    save_json(params)
