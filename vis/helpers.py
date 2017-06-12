import os
import json
import sys
import pandas
sys.path.append("./../sysinfo")
from cute_device import get_cute_device_str


def read_file(filename):
    with open(filename) as f:
        data = json.load(f)
    return data


def read_df_from_dir(path):
    data = [read_file(os.path.join(path, f)) for f in os.listdir(path) if not f.startswith("arch") and os.path.isfile(os.path.join(path, f))]
    df = pandas.DataFrame(data)
    df["device"] = df["device"].apply(get_cute_device_str)
    return df


def filter_by(df, filters):
    df_plot = df
    for key in filters:
        df_plot = df_plot[df_plot[key] == filters[key]]
    return df_plot
