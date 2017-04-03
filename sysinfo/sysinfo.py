import begin
import json
import os
import tempfile
import subprocess


def get_script_dir():
    path = os.path.dirname(os.path.realpath(__file__))
    return path


def get_sys_info():
    script_path = os.path.join(get_script_dir(), '_sysinfo.py')
    with tempfile.TemporaryFile() as tempf:
        proc = subprocess.Popen(script_path, stdout=tempf)
        proc.wait()
        tempf.seek(0)
        b_info= (tempf.read())
    json_info=b_info.decode()
    dic_info = json.loads(json_info)
    return dic_info

@begin.start
def main():
    info = get_sys_info()
    print (json.dumps(info, sort_keys=True,  indent=4, separators=(',', ': ')))
