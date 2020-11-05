import subprocess

from benchmarker.util import abstractprocess


def get_nvprof_counters(command):
    sep = "flop_count_sp   Floating Point Operations(Single Precision)"
    nvprof_command = [
        "nvprof",
        "--profile-child-processes",
        "--metrics",
        "flop_count_sp",
    ]
    proc = abstractprocess.Process("local", command=nvprof_command + command)
    process_err = proc.get_output()["err"]
    gflop_measured = 0.0
    for line in process_err.split("\n"):
        if sep in line:
            cnt, mma = line.split(sep)
            cnt = cnt.strip()
            minimum, maximum, average = mma.split()
            gflop_measured += float(cnt) * float(average)

    return gflop_measured / 10 ** 9
