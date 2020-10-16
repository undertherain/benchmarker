def detalize_ops_results(results):
    if "gflop_estimated" in results["problem"]:
        results["gflop_per_second"] = results["problem"]["gflop_estimated"] / results["time_total"]
    if results["power"]["joules_total"] > 0:
        results["gflop_per_joule"] = results["problem"]["gflop_estimated"] / results["power"]["joules_total"]
    if "joules_CPU" in results["power"]:
        results["gflop_per_joule_CPU"] = results["problem"]["gflop_estimated"] / results["power"]["joules_CPU"]
        if results["nb_gpus"] == 0:
            results["gflop_per_joule_device"] = results["gflop_per_joule_CPU"]
        results["gflop_per_joule_device"] = results["gflop_per_joule_CPU"]
    if "joules_GPU" in results["power"]:
        results["gflop_per_joule_GPU"] = results["problem"]["gflop_estimated"] / results["power"]["joules_GPU"]
        results["gflop_per_joule_device"] = results["gflop_per_joule_GPU"]
