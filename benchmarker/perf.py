import re
import subprocess

perf_counters_multipliers = {'r5302c7': 1,
                             'r5308c7': 4,
                             'r5320c7': 8}


# TODO: reuse abstract process
def run(command):
    print(command)
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc.wait()
    out = proc.stdout.read().decode()
    err = proc.stderr.read().decode()
    if proc.returncode == 0:
        return [out, err]
    else:
        raise RuntimeError(f"could not run perf: {err}")


def get_counters(output_dict, command):
    output_dict["problem"]["flop_measured"] = 0
    for counter in perf_counters_multipliers:
        perf_command = ["perf", "stat", "-e", counter]
        output_string, error_string = run(perf_command + command)
        print(error_string)
        if error_string:
            match_exp = re.compile('[\d|\,]+\s+' + counter).search(error_string)
            if match_exp:
                match_list = match_exp.group().split()
                cntr_value = int(match_list[0].replace(',',''))
                output_dict["problem"]["flop_measured"] += perf_counters_multipliers[counter] * cntr_value
    return output_dict
