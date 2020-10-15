#!/usr/bin/env python3
import pandas
import os
import time
import json
import sys
from benchmarker.util.io import save_json
from benchmarker.util.cute_device import get_cute_device_str

batchsize = [32]
epoch = 10 


def main():
    if len(sys.argv) != 2:
        sys.exit("Enter csv file!!!")
    params = {}
    params["framework"] = "pytorch"
    params["mode"] = "inference"

    specs = pandas.read_csv(sys.argv[1])
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




if __name__ == "__main__":
    print(sys.path)
    main()
