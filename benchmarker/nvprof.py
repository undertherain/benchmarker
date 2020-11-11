from benchmarker.util import abstractprocess


def get_nvprof_counters(command, precision):
    if precision == "FP16":
        metric = "flop_count_hp"
        sep_str = "Half"
    elif precision == "FP32":
        metric = "flop_count_sp"
        sep_str = "Single"
    sep = metric + "   Floating Point Operations(" + sep_str + " Precision)" 
    nvprof_command = [
        "nvprof",
        "--profile-child-processes",
        "--metrics",
        metric,
    ]
    proc = abstractprocess.Process("local", command=nvprof_command + command)
    process_err = proc.get_output()["err"]
    print(process_err)
    gflop_measured = 0.0
    for line in process_err.split("\n"):
        if sep in line:
            cnt, mma = line.split(sep)
            cnt = cnt.strip()
            minimum, maximum, average = mma.split()
            gflop_measured += float(cnt) * float(average)

    return gflop_measured / 10 ** 9
