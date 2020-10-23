import re
import subprocess

from benchmarker.util import abstractprocess


def get_counters(command):
    flop_measured = 0
    for counter in perf_counters_multipliers:
        for rep in [1, 8]:
            fapp_output = "./prof"
            csv_output = "./csvs/"
            # fmt: off
            # fapp -C -d ./prof_${APP}_rep${REP} -Hevent=pa${REP} ./${APP}
            fapp_measure_cmd = ["fapp", "-C", "-d", f"./prof_rep{rep} -Hevent=pa{rep}"]
            # fapp -A -tcsv -o ${APP}_reps/pa$REP.csv -d ./prof_${APP}_rep${REP} -Icpupa
            fapp_gen_csv_cmd = ["fapp", "-A", "-tcsv", "-o", "reps/pa{rep}.csv", "-d", "./prof_rep{rep}", "-Icpupa",]
            # fmt: on
            proc = abstractprocess.Process("local", command=fapp_measure_cmd + command)

            # process_err = proc.get_output()["err"]
            # print(process_err)
        # delete tmp files
        match_exp = re.compile("[\d|\,]+\s+" + counter).search(process_err)
        match_list = match_exp.group().split()
        cntr_value = int(match_list[0].replace(",", ""))
        flop_measured += perf_counters_multipliers[counter] * cntr_value
    gflop_measured = flop_measured / 10 ** 9
    return gflop_measured
