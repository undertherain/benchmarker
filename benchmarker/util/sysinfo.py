import json
import logging
import os
import sys

from benchmarker.util import abstractprocess, get_script_dir


def get_sys_info():
    logger = logging.getLogger(__name__)
    script_path = os.path.join(get_script_dir(), "_sysinfo.py")
    script_cmd = [sys.executable, script_path]
    proc = abstractprocess.Process("local", command=script_cmd)
    result = proc.get_output()
    if result["returncode"] != 0:
        logger.error("Cannot get system info")
        # print(result)
        return {}
    json_info = result["out"]
    dic_info = json.loads(json_info)
    return dic_info


def main():
    info = get_sys_info()
    print(json.dumps(info, sort_keys=True, indent=4, separators=(",", ": ")))


if __name__ == "__main__":
    main()
