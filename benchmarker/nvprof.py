import re
import subprocess
from benchmarker.util import abstractprocess

nvprof_expr1_re = re.compile('(\d+)\s+flop\_count\_sp\s+Floating\s+Point\s+Operations\(Single Precision\)\s+(?:(\d+)\s+){3}')
nvprof_expr2_re = re.compile('(\d+)\s+flop\_count\_sp\s+Floating\s+Point\s+Operations\(Single Precision\)\s+(?:(\d+\.\d+e[\+|\-]\d+)\s+){3}')


def get_nvprof_counters(command):
    nvprof_command = ["nvprof", "--profile-child-processes", "--metrics", 'flop_count_sp']
    proc = abstractprocess.Process("local", command=nvprof_command + command)
    process_err = proc.get_output()["err"]
    #print(process_err)
    match_expr1 = nvprof_expr1_re.findall(process_err)
    #print(match_expr1)
    output_expr1 = [int(x) * int(y) for x, y in match_expr1]
    match_expr2 = nvprof_expr2_re.findall(process_err)
    #print(match_expr2)
    output_expr2 = [int(x) * int(float(y)) for x, y in match_expr2]
    output = output_expr1 + output_expr2
    #print(output)
    gflop_measured = sum(output) / 10 ** 9
    return gflop_measured

