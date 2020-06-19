# -*- coding: utf-8 -*-
"""Benchmarker main module

This is where all magic is happening
"""

import argparse
import ast
import importlib
import pkgutil
# import logging
import os
from .util import sysinfo
from .util.io import save_json
from benchmarker.util.cute_device import get_cute_device_str


def get_modules():
    path_modules = "benchmarker/modules"
    return [name for _, name, is_pkg in pkgutil.iter_modules([path_modules])
            if not is_pkg and name.startswith('do_')]


def parse_basic_args(argv):
    parser = argparse.ArgumentParser(description='Benchmark me up, Scotty!')
    parser.add_argument("--framework")
    parser.add_argument("--problem")
    parser.add_argument('--path_out', type=str, default="./logs")
    parser.add_argument('--gpus', default="")
    parser.add_argument('--problem_size', default=None)
    parser.add_argument('--batch_size', default=None)
    # parser.add_argument('--misc')
    return parser.parse_known_args(argv)


def run(argv):
    args, unknown_args = parse_basic_args(argv)
    params = {}
    params["platform"] = sysinfo.get_sys_info()
    if args.framework is None:
        print("please choose one of the frameworks to evaluate")
        print("available frameworks:")
        for plugin in get_modules():
            print("\t", plugin[3:])
        raise Exception

    # TODO: load frameowork's metadata from backend
    # TODO: make framework details nested
    params["framework"] = args.framework
    params["path_out"] = args.path_out

    if args.problem is None:
        print("choose a problem to run")
        print("problems supported by {}:".format(args.framework))
        raise Exception
    # TODO: get a list of support problems for a given framework

    # TODO: load problem's metadata from the problem itself
    params["problem"] = {}
    params["problem"]["name"] = args.problem
    # TODO: move this to the root base benchmark
    if args.problem_size is not None:
        params["problem"]["size"] = ast.literal_eval(args.problem_size)
    if args.batch_size is not None:
        params["batch_size"] = int(args.batch_size)
        params["batch_size_per_device"] = int(args.batch_size)
    # params["misc"] = args.misc
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        params["gpus"] = list(map(int, args.gpus.split(",")))
    else:
        params["gpus"] = []

    params["nb_gpus"] = len(params["gpus"])

    if params["nb_gpus"] > 0:
        params["device"] = params["platform"]["gpus"][0]["brand"]
    else:
        if params["platform"]["cpu"]["brand"] is not None:
            params["device"] = params["platform"]["cpu"]["brand"]
        else:
            # TODO: add arch when it becomes available thougg sys query
            params["device"] = "unknown CPU"

    params["path_out"] = os.path.join(params["path_out"], params["problem"]["name"])
    mod = importlib.import_module("benchmarker.modules.do_" + params["framework"])
    benchmark = getattr(mod, 'Benchmark')(params, unknown_args)
    benchmark.run()
    cute_device = get_cute_device_str(params["device"]).replace(" ", "_")
    params["path_out"] = os.path.join(params["path_out"], cute_device)
    save_json(params)
