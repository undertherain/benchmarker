# -*- coding: utf-8 -*-
"""Benchmarker main module

This is where all magic is happening
"""

import argparse
import ast
import importlib
import os
import pkgutil
import sys

from .util.io import print_json


def get_modules():
    path_modules = "benchmarker/modules"
    return [
        name
        for _, name, is_pkg in pkgutil.iter_modules([path_modules])
        if not is_pkg and name.startswith("do_")
    ]


def parse_basic_args(argv):
    parser = argparse.ArgumentParser(description="Benchmark me up, Scotty!")
    parser.add_argument("--framework")
    parser.add_argument("--problem")
    parser.add_argument("--path_out", type=str, default="./logs")
    parser.add_argument("--gpus", default="")
    parser.add_argument("--problem_size", default=None)
    parser.add_argument("--power_sampling_ms", type=int, default=100)
    parser.add_argument("--preheat", action="store_true")

    # parser.add_argument('--misc')
    return parser.parse_known_args(argv)


def run(argv):
    args, unknown_args = parse_basic_args(argv)
    params = {}
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

    # TODO: load problem's metadata  from the problem itself
    params["preheat"] = args.preheat
    params["problem"] = {}
    params["problem"]["name"] = args.problem
    # TODO: move this to the root base benchmark
    if args.problem_size is not None:
        params["problem"]["size"] = ast.literal_eval(args.problem_size)
    # params["misc"] = args.misc
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        params["gpus"] = list(map(int, args.gpus.split(",")))
    else:
        params["gpus"] = []

    params["nb_gpus"] = len(params["gpus"])

    params["power"] = {}
    params["power"]["sampling_ms"] = args.power_sampling_ms
    params["power"]["joules_total"] = 0
    params["power"]["avg_watt_total"] = 0
    mod = importlib.import_module("benchmarker.modules.do_" + params["framework"])
    benchmark = getattr(mod, "Benchmark")(params, unknown_args)
    # TODO: make this optional
    benchmark.measure_power_and_run()
    print("benchmarkmagic#!%")
    print_json(params)


if __name__ == "__main__":
    run(sys.argv[1:])
