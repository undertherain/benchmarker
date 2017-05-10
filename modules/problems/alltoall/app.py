import os
import sys
import subprocess

''' IDEA: we can create an Alltoall superclass for other alltoall 
	implementations (Intel Micro Benchmark (IMB) etc.)
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
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate()

        if len(stderr) == 0:
            val = {} 
            val["timing"] = self.__parse_output(stdout.splitlines()[3:])
            val["config"] = cmd
        else:
            print(cmd)
            print("ERROR:")
            print(stderr)
            exit(1)

        return val

    def __compile(self, params=None):
        pass

    def __parse_output(self, output):
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

