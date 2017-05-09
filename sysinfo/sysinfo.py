import begin
import json
import os
import sys

sys.path.append("/myhome/home/kevin/workspace/github/benchmarker/util")
import abstractprocess

def get_script_dir():
    path = os.path.dirname(os.path.realpath(__file__))
    return path


def get_sys_info():
    script_path = os.path.join(get_script_dir(), '_sysinfo.py')
    proc = abstractprocess.Process("local", command = script_path)
    result = proc.get_output()
    if result["returncode"] != 0:
        exit(1)
    json_info = result["out"]
    dic_info = json.loads(json_info)
    return dic_info


@begin.start
def main():
    info = get_sys_info()
    print(json.dumps(info, sort_keys=True,  indent=4, separators=(',', ': ')))
