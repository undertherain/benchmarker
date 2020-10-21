import re
import subprocess
from benchmarker.util import abstractprocess

flops_as_integer = re.compile('(\d+)\s+flop\_count\_sp\s+Floating\s+Point\s+Operations\(Single Precision\)\s+(?:(\d+)\s+){3}')
flops_in_scientific_notation = re.compile('(\d+)\s+flop\_count\_sp\s+Floating\s+Point\s+Operations\(Single Precision\)\s+(?:(\d+\.\d+e[\+|\-]\d+)\s+){3}')


def get_nvprof_counters(command):
    nvprof_command = ["nvprof", "--profile-child-processes", "--metrics", 'flop_count_sp']
    proc = abstractprocess.Process("local", command=nvprof_command + command)
    process_err = proc.get_output()["err"]
    # print(process_err)
    matched_flops_int = flops_as_integer.findall(process_err)
    # print(matched_flops_int)
    total_flops_int = [int(x) * int(y) for x, y in matched_flops_int]
    matched_flops_scientific_notation = flops_in_scientific_notation.findall(process_err)
    # print(matched_flops_scientific_notation)
    total_flops_scientific_notation = [int(x) * int(float(y)) for x, y in matched_flops_scientific_notation]
    output = total_flops_int + total_flops_scientific_notation
    # print(output)
    gflop_measured = sum(output) / 10 ** 9
    return gflop_measured

