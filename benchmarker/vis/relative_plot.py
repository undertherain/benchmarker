import sys
from pathlib import Path

import pandas as pd
from benchmarker.util.cute_device import get_cute_device_str
from matplotlib import pyplot as plt
from protonn.vis import PivotTable, df_from_dir, filter_by


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import colorsys

    import matplotlib.colors as mc

    try:
        c = mc.cnames[color]
    except Exception:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def refactor_this(input_dir):
    plt.rcParams["figure.figsize"] = [10, 5]
    # mpl.style.use("dark_background")

    df_original = df_from_dir(Path(input_dir))
    # pd.set_option("display.max_rows", None, "display.max_columns", None)
    df_original["device"] = df_original["device"].apply(get_cute_device_str)
    df_original["problem.precision"] = df_original["problem.precision"].fillna("FP32")
    df_original["problem.precision"] = df_original["problem.precision"].apply(
        lambda x: x.replace("FP16", "mixed")
    )

    print(df_original["problem.precision"])
    df_original.drop("time", axis="columns", inplace=True)
    df_original.drop("path_out", axis="columns", inplace=True)
    # df_original.drop("GFLOP", axis="columns", inplace=True)
    df_original.drop("nb_gpus", axis="columns", inplace=True)

    for c in df_original.columns:
        if c.startswith("platform"):
            df_original.drop(c, axis="columns", inplace=True)

    df_original["device/backend"] = df_original.apply(
        lambda x: (
            (
                f"{x['device']}"
                # f"/{x['backend']}"
                f"/{x['problem.precision']}"
            )
        ),
        axis=1,
    )
    # Dataframe to plot relative performance of each device/backend
    # print(df_original)
    # exit(0)
    df_plot = pd.DataFrame()
    for problem in df_original["problem.name"].unique():
        print("\n\n########################\n")
        print(problem)
        filters = {"problem.name": problem}
        df_new = filter_by(df_original, filters)
        df_new["problem.name"] = problem
        if len(df_new) == 0:
            continue
        print(df_new.columns)
        dev_baseline = "Xeon E5-2650 v4/FP32"  # "Xeon Gold 6148"
        df_new = df_new.groupby(["device/backend"]).max()
        df_new.reset_index(inplace=True)
        print("new df:")
        print(df_new)
        if problem == "gemm":
            perf_baseline = df_new[df_new["device/backend"] == dev_baseline][
                "GFLOP/sec"
            ].max()
            df_new["perf_relative"] = df_new["GFLOP/sec"] / perf_baseline
        else:
            perf_baseline = df_new[df_new["device/backend"] == dev_baseline][
                "samples_per_second"
            ].max()
            df_new["perf_relative"] = df_new["samples_per_second"] / perf_baseline
        df_plot = df_plot.append(df_new)

    df_plot.reset_index(inplace=True)
    print("DF plot------------------")
    print(df_plot)
    # exit(0)
    key_target = "perf_relative"
    pt = PivotTable(
        key_target=key_target,
        key_primary="device/backend",
        key_secondary="problem.name",
        keys_maximize=[],
    )

    df_mean, df_max, df_std = pt.pivot_dataframe(df_plot)
    # fig = plt.figure(figsize=(30, 9)) # dunno why this is not working
    print(df_mean)
    df_mean = df_mean.reindex(["gemm", "bert", "resnet50", "vgg16", "ncf"])
    # df_mean.plot.bar(yerr=df_std)
    colors = [
        "r",
        lighten_color("r"),
        "g",
        lighten_color("g"),
        "b",
        lighten_color("b"),
        "orange",
        lighten_color("orange"),
    ]
    df_mean.plot.bar(color=colors)
    plt.ylabel("relative perofmance to Xeon 2650")
    # figname = "batch" + str(batchsize) + "_" + conv_type
    # plt.title(conv_type)
    # name_file_out = key_target + "_" + ",".join([f"{k}_{filters[k]}" for k in filters]) + "_bars.pdf"
    name_file_out = key_target + "_" + "mlperf" + "_bars.pdf"
    plt.savefig(name_file_out, bbox_inches="tight", transparent=False)

    plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"run {sys.argv[0]} path_to_logs")
        exit(-1)

    refactor_this(sys.argv[1])
