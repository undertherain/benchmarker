#!/usr/bin/env python3
import subprocess
import tempfile


def run(params):
    out_file = tempfile.TemporaryFile()
    err_file = tempfile.TemporaryFile()
    command = ["python3", "-m", "benchmarker"]
    for key in params:
        command.append(f"--{key}")
        command.append(str(params[key]))
    # print(command)
    proc = subprocess.Popen(command, stdout=out_file, stderr=err_file)
    proc.wait()
    out_file.seek(0)
    out = out_file.read().decode()
    err_file.seek(0)
    err = err_file.read().decode()
    out_file.close()
    err_file.close()
    result = {"returncode": proc.returncode, "out": out, "err": err}
    # print(result)
    if result["returncode"] != 0:
        print(result["err"])
