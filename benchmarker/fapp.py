from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarker.fapp_power import get_power, get_total_power
from benchmarker.util import abstractprocess


def run_fapp_profiler(fapp_dir, command):
    # fapp -C -d ./prof_${APP}_rep${REP} -Hevent=pa${REP} ./${APP}
    fapp_measure_cmd = ["fapp", "-C", "-d"]
    fapp_measure_cmd += [fapp_dir, "-Hevent=pa{rep}"]
    cmd = fapp_measure_cmd + command
    print(cmd)
    abstractprocess.Process("local", command=cmd)


def gen_fapp_csv(fapp_dir, csv_file):
    # fapp -A -tcsv -o $APP_reps/pa$REP.csv -d ./prof_$APP_rep$REP -Icpupa
    fapp_gen_csv_cmd = ["fapp", "-A", "-tcsv"]
    fapp_gen_csv_cmd += ["-o", csv_file] + ["-d", fapp_dir, "-Icpupa"]
    print(fapp_gen_csv_cmd)
    abstractprocess.Process("local", command=fapp_gen_csv_cmd)


def get_power_total_and_detail(command):
    csv_dir = Path("csvs")
    csv_dir.mkdir()
    for rep in [1, 8]:
        with TemporaryDirectory(suffix=str(rep)) as fapp_dir:
            csv_file = str(csv_dir.joinpath(f"pa{rep}.csv"))
            run_fapp_profiler(fapp_dir, command)
            gen_fapp_csv(fapp_dir, csv_file)
        # delete fapp_dir

    power_details = get_power(csv_dir)
    power_total = get_total_power(power_details)
    csv_dir.rmdir()

    return power_total, power_details
