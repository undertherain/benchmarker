import os
from tempfile import TemporaryDirectory

from benchmarker.profiling.power_fapp import get_power, get_total_power
from benchmarker.util import abstractprocess

from ..util.io import get_path_out_dir, get_path_out_name


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


def get_power_total_and_detail(command, params):
    prefix = get_path_out_dir(params)
    suffix_data = get_path_out_name(params, "fapp_data")
    suffix_csv = get_path_out_name(params, "fapp_csv")
    fapp_dir = os.path.join(prefix, suffix_data)
    csv_dir = os.path.join(prefix, suffix_csv)
    os.makedirs(fapp_dir)
    os.makedirs(csv_dir)

    print("COMMAND:", command)
    print("FAPP_DIR:", fapp_dir)
    print("CSV_DIR:", csv_dir)
    for rep in [1, 8]:
        csv_file = f"{csv_dir}/pa{rep}.csv"
        run_fapp_profiler(fapp_dir, rep, command)
        gen_fapp_csv(fapp_dir, csv_file)
    power_details = get_power(csv_dir)
    power_total = get_total_power(power_details)

    return power_total, power_details
