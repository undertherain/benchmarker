import re
import subprocess
from benchmarker.util import abstractprocess

exp1_re = re.compile('(\d+)\s+flop\_count\_sp\s+Floating\s+Point\s+Operations\(Single Precision\)\s+(?:(\d+)\s+){3}')
exp2_re = re.compile('(\d+)\s+flop\_count\_sp\s+Floating\s+Point\s+Operations\(Single Precision\)\s+(?:(\d+\.\d+e[\+|\-]\d+)\s+){3}')


def get_nvprof_counters(command):
    nvprof_command = ["nvprof", "--profile-child-processes", "--metrics", 'flop_count_sp']
    proc = abstractprocess.Process("local", command=nvprof_command + command)
    process_err = proc.get_output()["err"]
    #print(process_err)
    match_exp1 = exp1_re.findall(process_err)
    #print(match_exp1)
    output_exp1 = [int(x) * int(y) for x, y in match_exp1]
    match_exp2 = exp2_re.findall(process_err)
    #print(match_exp2)
    output_exp2 = [int(x) * int(float(y)) for x, y in match_exp2]
    output = output_exp1 + output_exp2
    #print(output)
    flop_measured = sum(output)
    return flop_measured

