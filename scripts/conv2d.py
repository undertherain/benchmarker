#!/usr/bin/env python3
import subprocess
import pandas
import os
import time
import json
import re

batchsize = [32]
epoch = 10 

perf_counters_multipliers = {'r5302c7': 1, 
                             'r5308c7': 4,
                             'r5320c7': 8}

def run(params, command):
    for key, val in params.items():
        command.append(f"--{key}={val}")
    print(command)
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc.wait()
    out = proc.stdout.read().decode()
    err = proc.stderr.read().decode()
    if proc.returncode == 0:
        output_dict = json.loads(out)
        return output_dict
    else:
        print(out)
        print(err)


def get_counters(params, output_dict):
    output_dict["flop_measured"] = 0
    for counter in perf_counters_multipliers:
        perf_command = ["perf", "stat" , "-e", counter, "python3", "-m", "benchmarker", "--no_cudnn_benchmark"]
        for key, val in params.items():
            perf_command.append(f"--{key}={val}")
        print(perf_command)
        proc = subprocess.Popen(perf_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        proc.wait()
        out = proc.stdout.read().decode()
        err = proc.stderr.read().decode()
        print(err)
        #if proc.returncode != 0:
        #print(out)
        #print(err)
        if err:
            match_exp = re.compile('[\d|\,]+\s+' + counter).search(err)
            if match_exp:
                match_list = match_exp.group().split()
                cntr_value = int(match_list[0].replace(',',''))
                output_dict["flop_measured"] += perf_counters_multipliers[counter]*cntr_value
    print(output_dict)

            

def main():
    params = {}
    params["framework"] = "pytorch"
    params["mode"] = "inference"

    specs = pandas.read_csv("./conv2d.csv")
    for size in batchsize: 
        params["batch_size"] = size
        for row in specs.iterrows():
            spec = row[1]
            params["problem"] = spec["conv_type"]
            prob_size = params["batch_size"] * 4

            if params["problem"] == 'conv2d':
                params["problem_size"] = (f"{prob_size},"
                                      f"{spec['in_channels']},"
                                      f"{spec['in_width']},"
                                      f"{spec['in_height']}")
        
            params["cnt_filters"] = spec["cnt_filters"]
            params["size_kernel"] = spec["width_kernel"]
            params["stride"] = spec["stride"]
            params["dilation"] = spec["dilation"]
            params["padding"] = spec["padding"]
            params["nb_epoch"] = epoch 
            #print(params)
         
            command = ["python3", "-m", "benchmarker", "--no_cudnn_benchmark"] 

            #Run benchmarker with params and get output as dict
            output_dict = run(params, command)
            print(output_dict)

            #Collect counters, add to the dict
            get_counters(params, output_dict)

if __name__ == "__main__":
    main()
