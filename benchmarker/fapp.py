import re
import subprocess
from tempfile import TemporaryDirectory

from benchmarker.util import abstractprocess


def run_fapp_profiler(fapp_dir, command):
    # fapp -C -d ./prof_${APP}_rep${REP} -Hevent=pa${REP} ./${APP}
    fapp_measure_cmd = ["fapp", "-C", "-d"]
    fapp_measure_cmd += [fapp_dir, "-Hevent=pa{rep}"]
    proc = abstractprocess.Process("local", command=fapp_measure_cmd + command)


def gen_fapp_csv(fap_dir, csv_file):
    # fapp -A -tcsv -o ${APP}_reps/pa$REP.csv -d ./prof_${APP}_rep${REP} -Icpupa
    fapp_gen_csv_cmd = ["fapp", "-A", "-tcsv"]
    fapp_gen_csv_cmd += ["-o", csv_file] + ["-d", fapp_dir, "-Icpupa"]
    proc = abstractprocess.Process("local", command=fapp_gen_csv_cmd)


def get_counters(command):
    flop_measured = 0
    for counter in perf_counters_multipliers:
        for rep in [1, 8]:
            fapp_dir = f"./prof{rep}"
            csv_file = f"./csvs/pa{rep}.csv"
            run_fapp_profiler(fapp_dir, command)
            gen_fapp_csv(fap_dir, csv_file)
            # process_err = proc.get_output()["err"]
            # print(process_err)
        # delete tmp files
        match_exp = re.compile("[\d|\,]+\s+" + counter).search(process_err)
        match_list = match_exp.group().split()
        cntr_value = int(match_list[0].replace(",", ""))
        flop_measured += perf_counters_multipliers[counter] * cntr_value
    gflop_measured = flop_measured / 10 ** 9
    return gflop_measured
