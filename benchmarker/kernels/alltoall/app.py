import os
import sys
import subprocess
from util import abstractprocess
''' IDEA: we can create an Alltoall superclass for other alltoall 
	implementations (Intel Micro Benchmark (IMB) etc.)

    This requires an alltoall microbenchmark.
        OSU Micro-Benchmark
         - http://mvapich.cse.ohio-state.edu/benchmarks/
'''

class OSU_Alltoall():
    def __init__(self, params):
        self.path        = os.path.dirname(os.path.realpath(__file__))
        self.source_code = "osu_alltoall.c"
        self.binary      = "osu_alltoall"
        self.run_args    = ["-f"]

        self.__compile(params)

    def add_run_args(self, params):
        if "message_sizes" in params:
            self.run_args.append("-m")
            self.run_args.append(params["message_sizes"])
        if "iterations" in params:
            self.run_args.append("-x")
            self.run_args.append(params["iterations"])

    def run_command(self, params=None):
        app = os.path.join(self.path, self.binary)
        return [app] + self.run_args

    def execute(self, mpi):
        cmd  = mpi + self.run_command()
        proc = abstractprocess.Process("local", command= cmd)
        output = proc.get_output()
        if output["returncode"] != 0:
            print("Cannot get benchmark. Exiting.") 
            print(output)
            exit(1)
        
        val = {}
        val["config"] = " ".join(cmd)
        val["timing"] = self.__parse_output(output["out"])

        return val

    def __compile(self, params=None):
        pass

    def __parse_output(self, output):
        output = output.splitlines()[3:]
        results = {}
        for line in output:
            size, t_avg, t_min, t_max, iterations = line.split()
            size = int(size)
            results[size] = {}
            results[size]["avg"] = float(t_avg)
            results[size]["min"] = float(t_min)
            results[size]["max"] = float(t_max)
            results[size]["iterations"] = int(iterations)

        return results

''' In principle, we should be able to get a binary that has been compiled using the 
	MPI library that we are benchmarking (Open MPI, MVAPICH, MPICH, etc)
'''
def get_app(params=None):
    app = OSU_Alltoall(params)

    return app

def main():
    app = get_app(None)
    print (app.run_command(None))

