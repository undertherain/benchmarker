#!/usr/bin/env python3
import subprocess
import tempfile
import pandas
import os
import time

basedir = "./flops"
batchsize = [32] 

def run(params, command, output_file):
    with open(output_file, "a") as f:
        out_file = f
        command.extend(("python3", "-m", "benchmarker", "--no_cudnn_benchmark"))
        for key in params:
            command.append(f"--{key}")
            command.append(str(params[key]))
        print(command)
        proc = subprocess.Popen(command, stdout=out_file, stderr=out_file)
        proc.wait()
        out_file.close()

def main():
    params = {}
    params["framework"] = "pytorch"
    params["mode"] = "inference"

    specs = pandas.read_csv("./conv2d_specs.csv")
    print(specs)
    for size in batchsize: 
        params["batch_size"] = size
        for row in specs.iterrows():
            spec = row[1]
            params["problem"] = spec["conv_type"]
            prob_size = params["batch_size"] * 4
            print(spec)

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
            #print (params)
           
            filename = (f"{params['problem']}"
			f"_in{spec['in_channels']}"
			f"x{spec['in_width']}"
			f"x{spec['in_height']}"
                        f"_k{params['cnt_filters']}"
                        f"_s{params['stride']}"
                        f"_p{params['padding']}"
                        f"_bs{params['batch_size']}")
          
            out_dir = os.path.join(basedir, params["problem"], "perf")
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)  
            output_file = os.path.join(out_dir, filename) 
            command = []
            params["backend"] = "DNNL"
            run(params, command, output_file)
            command = ["perf", "stat", "-e", "r5301c7", "-e", "r5302c7", "-e", "r5304c7", "-e", "r5308c7", "-e", "r5310c7", "-e", "r5320c7"] 
            run(params, command, output_file)


if __name__ == "__main__":
    main()
