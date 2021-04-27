# -*- coding: utf-8 -*-
"""Benchmarker main module

This is where all magic is happening
"""

import argparse
import ast
import os
import pkgutil
import sys
from importlib import import_module

from .util.io import print_json


def parse_basic_args(argv):
    parser = argparse.ArgumentParser(description="Benchmark me up, Scotty!")
    parser.add_argument("--framework")
    parser.add_argument("--problem")
    parser.add_argument("--problem_size", default=None)
    parser.add_argument("--path_out", type=str, default="./logs")
    parser.add_argument("--gpus", default="")
    parser.add_argument("--power_rapl", action="store_true", default=False)
    parser.add_argument("--power_sampling_ms", type=int, default=100)
    parser.add_argument("--preheat", action="store_true")

    # parser.add_argument('--misc')
    return parser.parse_known_args(argv)


def get_framework(args):
    # TODO: load frameowork's metadata from backend
    # TODO: make framework details nested
    if args.framework is None:
        print("please choose one of the frameworks to evaluate")
        print("available frameworks:")
        candidates = pkgutil.iter_modules(["benchmarker/frameworks"])
        for _, name, is_pkg in candidates:
            if not is_pkg and name.startswith("do_"):
                print(f"\t{name[3:]}")
        raise ValueError("No framework specified")
    return args.framework


def get_problem(args):
    if args.problem is None:
        # TODO: get a list of support problems for a given framework
        # print("choose a kernel/problem to run")
        # print("problems supported by {}:".format(args.framework))
        print("Exact listing not implemented!")
        print("Look into the `benchmarker/kernels` dir!")
        raise ValueError("No kernel/problem specified")

    # TODO: load problem's metadata  from the problem itself
    problem = {}
    problem["name"] = args.problem
    # TODO: move this to the root base benchmark
    if args.problem_size is not None:
        problem["size"] = ast.literal_eval(args.problem_size)
    return problem


def get_gpus(args):
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        return list(map(int, args.gpus.split(",")))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        return []


def run(argv):
    args, unknown_args = parse_basic_args(argv)
    params = {}
    params["framework"] = get_framework(args)
    params["path_out"] = args.path_out
    # params["env"] = dict(os.environ)
    # params["misc"] = args.misc
    params["preheat"] = args.preheat
    params["problem"] = get_problem(args)
    params["gpus"] = get_gpus(args)
    params["nb_gpus"] = len(params["gpus"])

    params["power"] = {}
    params["power"]["sampling_ms"] = args.power_sampling_ms
    params["power"]["joules_total"] = 0
    params["power"]["avg_watt_total"] = 0
    framework_mod = import_module(f"benchmarker.frameworks.do_{params['framework']}")
    benchmark = framework_mod.Benchmark(params, unknown_args)
    # TODO: make this optional
    if args.power_rapl:
        benchmark.measure_power_and_run()
    print("benchmarkmagic#!%")
    print_json(params)


if __name__ == "__main__":
    run(sys.argv[1:])
