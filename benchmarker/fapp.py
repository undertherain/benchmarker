from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarker.util import abstractprocess
from fapp_power import get_power, get_total_power


def run_fapp_profiler(fapp_dir, command):
    # fapp -C -d ./prof_${APP}_rep${REP} -Hevent=pa${REP} ./${APP}
    fapp_measure_cmd = ["fapp", "-C", "-d"]
    fapp_measure_cmd += [fapp_dir, "-Hevent=pa{rep}"]
    abstractprocess.Process("local", command=fapp_measure_cmd + command)


def gen_fapp_csv(fapp_dir, csv_file):
    # fapp -A -tcsv -o $APP_reps/pa$REP.csv -d ./prof_$APP_rep$REP -Icpupa
    fapp_gen_csv_cmd = ["fapp", "-A", "-tcsv"]
    fapp_gen_csv_cmd += ["-o", csv_file] + ["-d", fapp_dir, "-Icpupa"]
    abstractprocess.Process("local", command=fapp_gen_csv_cmd)


def get_counters(command):
    csv_dir = Path("csvs")
    for rep in [1, 8]:
        with TemporaryDirectory(suffix=str(rep)) as fapp_dir:
            csv_file = csv_dir.joinpath(f"pa{rep}.csv")
            run_fapp_profiler(fapp_dir, command)
            gen_fapp_csv(fapp_dir, csv_file)
        # delete fapp_dir

    power = get_power(csv_dir)
    total_power = get_total_power(power)
    csv_dir.rmdir()

    return gflop_measured
