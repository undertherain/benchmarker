from benchmarker.util.io import print_json


def add_power_details(dic_power, time_total):
    # dic_power["joules_total"] = 0
    # dic_power["avg_watt_total"] = 0

    for sufix in ["CPU", "RAM", "GPU", "total"]:
        key_joules = f"joules_{sufix}"
        key_watt = f"avg_watt_{sufix}"

        if key_watt not in dic_power:
            if key_joules in dic_power:
                dic_power[key_watt] = dic_power[key_joules] / time_total

        if key_joules not in dic_power:
            if key_watt in dic_power:
                dic_power[key_joules] = dic_power[key_watt] * time_total

    if dic_power["joules_total"] == 0:
        for sufix in ["CPU", "RAM", "GPU"]:
            key_joules = f"joules_{sufix}"
            key_watt = f"avg_watt_{sufix}"
            if key_joules in dic_power:
                dic_power["joules_total"] += dic_power[key_joules]
            if key_watt in dic_power:
                dic_power["avg_watt_total"] += dic_power[key_watt]


def add_flops_details(results):
    problem = results["problem"]
    power = results["power"]
    if "gflop_measured" in problem:
        gflops = float(problem["gflop_measured"])
    elif "gflop_estimated" in problem:
        gflops = float(problem["gflop_estimated"])
    # TODO: EMIL VE VERY SMART ABOUT IT
    if "gflops" in locals():
        results["gflop_per_second"] = gflops / results["time_total"]
        if results["power"]["joules_total"] != 0:
            results["gflop_per_joule"] = gflops / float(power["joules_total"])

        if "joules_CPU" in results["power"]:
            results["gflop_per_joule_CPU"] = gflops / power["joules_CPU"]
            if results["nb_gpus"] == 0:
                results["gflop_per_joule_device"] = results["gflop_per_joule_CPU"]

        if "joules_GPU" in results["power"]:
            results["gflop_per_joule_GPU"] = gflops / power["joules_GPU"]
            results["gflop_per_joule_device"] = results["gflop_per_joule_GPU"]


def add_result_details(result):
    add_power_details(result["power"], result["time_total"])
    add_flops_details(result)
    # print_json(result)
    if result["power"]["joules_total"] > 0 and "cnt_samples" in result["problem"]:
        result["samples_per_joule"] = result["problem"]["cnt_samples"] * \
            result["nb_epoch"] / result["power"]["joules_total"]
