from pathlib import Path
from protonn.vis import df_from_dir, PivotTable, filter_by
from matplotlib import pyplot as plt
from benchmarker.util.cute_device import get_cute_device_str
import pandas as pd
import sys
import matplotlib as mpl
from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = [10, 5]
# mpl.style.use("dark_background")

if len(sys.argv) < 2:
    print(f"run {sys.argv[0]} path_to_logs")
    exit(-1)

df_original = df_from_dir(Path(sys.argv[1]))
#pd.set_option("display.max_rows", None, "display.max_columns", None)
df_original["device"] = df_original["device"].apply(get_cute_device_str)
#print(df_original["device"])
df_original.drop("time", axis="columns", inplace=True)
df_original.drop("path_out", axis="columns", inplace=True)
# df_original.drop("GFLOP", axis="columns", inplace=True)
df_original.drop("nb_gpus", axis="columns", inplace=True)
for c in df_original.columns:
    if c.startswith("platform"):
        df_original.drop(c, axis="columns", inplace=True)


df_original["device/backend"] = df_original.apply(lambda x: (
        (f"{x['device']}"
         f"/{x['backend']}")), axis=1)

#Dataframe to plot relative performance of each device/backend
df_plot= pd.DataFrame()
for problem in df_original["problem.name"].unique():
    filters = {"problem.name": problem}
    df_new = filter_by(df_original, filters)
    df_new["problem.name"] = problem
    if len(df_new) == 0:
        continue
    print(df_new.columns)
    dev_baseline = "Xeon E5-2650 v4"  # "Xeon Gold 6148"
    df_new = df_new.groupby(["device"]).max()
    df_new.reset_index(inplace=True)
    #   print(df_new)
    perf_baseline = df_new[df_new["device"] == dev_baseline]["samples_per_second"].max()
    df_new["perf_relative"] = df_new["samples_per_second"] / perf_baseline
    df_plot = df_plot.append(df_new)

print(df_plot)
key_target = "perf_relative"
pt = PivotTable(key_target=key_target,
                key_primary="device",
                key_secondary="problem.name",
                keys_maximize=[])

df_mean_conv, df_max_conv, df_std_conv = pt.pivot_dataframe(df_plot)
# fig = plt.figure(figsize=(30, 9)) # dunno why this is not working
df_mean_conv.plot.bar(yerr=df_std_conv)
plt.ylabel("relative perofmance to Xeon 2650")
#figname = "batch" + str(batchsize) + "_" + conv_type
#plt.title(conv_type)
#name_file_out = key_target + "_" + ",".join([f"{k}_{filters[k]}" for k in filters]) + "_bars.pdf"
name_file_out= key_target + "_" + "mlperf" + "_bars.pdf"
plt.savefig(name_file_out, bbox_inches="tight", transparent=False)

plt.close()
