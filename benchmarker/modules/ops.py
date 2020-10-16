def detalize_ops_results(results):
    problem = results["problem"]
    if "gflop_estimated" in results["problem"]:
        results["gflop_per_second"] = problem["gflop_estimated"] / results["time_total"]

        if results["power"]["joules_total"] > 0:
            results["gflop_per_joule"] = problem["gflop_estimated"] / results["power"]["joules_total"]

        if "joules_CPU" in results["power"]:
            results["gflop_per_joule_CPU"] = problem["gflop_estimated"] / results["power"]["joules_CPU"]
            if results["nb_gpus"] == 0:
                results["gflop_per_joule_device"] = results["gflop_per_joule_CPU"]

        if "joules_GPU" in results["power"]:
            results["gflop_per_joule_GPU"] = problem["gflop_estimated"] / results["power"]["joules_GPU"]
            results["gflop_per_joule_device"] = results["gflop_per_joule_GPU"]
