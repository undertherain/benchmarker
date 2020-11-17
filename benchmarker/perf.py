import re
import subprocess
from benchmarker.util import abstractprocess

perf_counters_multipliers = {'r5302c7': 1,
                             'r5308c7': 4,
                             'r5320c7': 8}

def get_gflop(command):
    flop_measured = 0
    for counter in perf_counters_multipliers:
        perf_command = ["perf", "stat", "-e", counter]
        proc = abstractprocess.Process("local", command=perf_command + command)
        process_err = proc.get_output()["err"]
        #print(process_err)
        match_exp = re.compile('[\d|\,]+\s+' + counter).search(process_err)
        match_list = match_exp.group().split()
        cntr_value = int(match_list[0].replace(',',''))
        flop_measured += perf_counters_multipliers[counter] * cntr_value
    gflop_measured = flop_measured / 10 ** 9
    return gflop_measured
