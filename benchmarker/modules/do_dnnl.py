import os
from benchmarker.util.abstractprocess import Process
# form a command line from parameters
path_benchdnn = "/home/blackbird/Projects_heavy/DL/mkl-dnn/build/tests/benchdnn"
spec_run = "mb64ic3ih224oc32oh224kh3ph1n\"myconv\""
command = [os.path.join(path_benchdnn, "./benchdnn"),
		   "--conv",
		   "--mode=p",
		   spec_run]
process = Process(command=command)
result = process.get_output()
std_out = result["out"]
lines = [line for line in std_out.split() if len(line) > 0]
time = lines[-1].split(":")[-1]
seconds = float(time) / 1000
print(seconds)
