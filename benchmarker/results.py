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
    else:
        return
    details = calculate_thoughput("gflop", gflops, power, results["time_total"])
    results.update(details)


def calculate_thoughput(metric_name, value, power, time):
    results = {}
    results[f"{metric_name}_per_second"] = value / time
    if power["joules_total"] != 0:
        results[f"{metric_name}_per_joule"] = value / float(power["joules_total"])
    if "joules_CPU" in power:
        results[f"{metric_name}_per_joule_CPU"] = value / power["joules_CPU"]
        if results["nb_gpus"] == 0:
            results[f"{metric_name}_per_joule_device"] = results[f"{metric_name}_per_joule_CPU"]
    if "joules_GPU" in power:
        results[f"{metric_name}_per_joule_GPU"] = value / power["joules_GPU"]
        results[f"{metric_name}_per_joule_device"] = results[f"{metric_name}_per_joule_GPU"]
    return results


def add_throughput_details(results):
    if results["power"]["joules_total"] > 0 and "cnt_samples" in results["problem"]:
    #    results["samples_per_joule"] = results["problem"]["cnt_samples"] * \
    #        results["nb_epoch"] / results["power"]["joules_total"]
        details = calculate_thoughput(
            "samples",
            results["problem"]["cnt_samples"] * results["nb_epoch"],
            results["power"],
            results["time_total"])
        results.update(details)


def add_result_details(result):
    add_power_details(result["power"], result["time_total"])
    add_flops_details(result)
    add_throughput_details(result)
    # print_json(result)
