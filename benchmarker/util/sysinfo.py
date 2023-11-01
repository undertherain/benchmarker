import json
import logging
import os
import sys

from benchmarker.util import abstractprocess, get_script_dir


def fixup_for_amd_gpus(result):
    if "gpus" not in result["platform"]:
        result["gpus"] = []
    same_gpu_count = len(result["platform"]["gpus"]) == result["nb_gpus"]
    using_pytorch = result["framework"] == "pytorch"
    if using_pytorch and not same_gpu_count:
        import torch.cuda

        nb_gpus = torch.cuda.device_count()
        gpus = []
        for i in range(nb_gpus):
            device_name = torch.cuda.get_device_name(i)
            # assert device_name == "Device 738c"
            # gpu = {"brand": "AMD Mi100"}
            gpus.append({"brand": device_name})

        result["platform"]["gpus"] = gpus


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
    # fixup_for_amd_gpus(dic_info)
    return dic_info


def main():
    info = get_sys_info()
    print(json.dumps(info, sort_keys=True, indent=4, separators=(",", ": ")))


if __name__ == "__main__":
    main()
