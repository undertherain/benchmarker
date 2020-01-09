from pathlib import Path
import json
import pandas
from pandas.io.json import json_normalize
from benchmarker.util.cute_device import get_cute_device_str


def read_file(filename):
    with open(filename) as f:
        data = json.load(f)
    return json_normalize(data)


def read_df_from_dir(path):
    path = Path(path)
    data = [read_file(f) for f in path.glob("**/*.json") if not f.name.startswith("arch")]
    if len(data) == 0:
        raise RuntimeError("no data to load")
    df = pandas.concat(data)
    df["device"] = df["device"].apply(get_cute_device_str)
    return df
