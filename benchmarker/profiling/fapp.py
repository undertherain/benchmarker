import argparse
from tempfile import TemporaryDirectory

from benchmarker.profiling.power_fapp import get_power, get_total_power
from benchmarker.util import abstractprocess


def get_path_out(command):
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_out")
    args = parser.parse_args(command)
    return args.path_out


def call_fapp(cmd):
    proc = abstractprocess.Process("local", command=cmd)
    output = proc.get_output()
    retval = output["returncode"]
    assert retval == 0, (
        "Error occurred with return code: {}\n"
        "Standard output:\n{}\n"
        "Standard output:\n{}\n"
    ).format(retval, output["out"], output["err"])


def run_fapp_profiler(fapp_dir, rep, command):
    # fapp -C -d ./prof_${APP}_rep${REP} -Hevent=pa${REP} ./${APP}
    fapp_measure_cmd = ["fapp", "-C", "-d"]
    fapp_measure_cmd += [fapp_dir, f"-Hevent=pa{rep}"]
    cmd = fapp_measure_cmd + command
    call_fapp(cmd)


def gen_fapp_csv(fapp_dir, csv_file):
    # fapp -A -tcsv -o $APP_reps/pa$REP.csv -d ./prof_$APP_rep$REP -Icpupa
    fapp_gen_csv_cmd = ["fapp", "-A", "-tcsv"]
    fapp_gen_csv_cmd += ["-o", csv_file] + ["-d", fapp_dir, "-Icpupa"]
    call_fapp(fapp_gen_csv_cmd)


def get_power_total_and_detail(command):
    print("PATH_OUT:", get_path_out(command))
    with TemporaryDirectory() as csv_dir:
        for rep in [1, 8]:
            with TemporaryDirectory(suffix=str(rep)) as fapp_dir:
                csv_file = f"{csv_dir}/pa{rep}.csv"
                run_fapp_profiler(fapp_dir, rep, command)
                gen_fapp_csv(fapp_dir, csv_file)
        power_details = get_power(csv_dir)
        power_total = get_total_power(power_details)

    return power_total, power_details
