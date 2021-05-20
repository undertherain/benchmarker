import csv
from pathlib import Path


def split_csv(pafile):
    with open(pafile) as csvfile:
        csvs = []
        row_len = 0
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) != row_len:
                csvs.append([])
                row_len = len(row)
            csvs[-1].append(row)
    return csvs


def get_df(csv):
    import pandas as pd

    return pd.DataFrame(csv[1:], columns=csv[0])


def get_freq(pa8):
    freq_df = get_df(pa8[0])
    name = "Timer clock frequency"
    freq = freq_df[freq_df.Name == name].Value.array[0]
    return float(freq)


def get_cores(pa8, cmg_id):
    cgm_df = get_df(pa8[2])
    cores = cgm_df[cgm_df.CMG == str(cmg_id)].Thread.array
    return cores


def get_tag(pafile, tag, region_name, cores):
    df = get_df(pafile[4])
    region_idcs = df["Region_name"] == region_name
    if cores.shape != (0,):
        cores_idcs = df.Thread.isin(cores)
    else:
        cores_idcs = df.Thread != "-"
    retdf = df[cores_idcs & region_idcs][tag]
    return retdf.astype("int64")


def get_core_power(core_counter, freq, cntvct):
    core_power = core_counter * freq * 8
    core_power /= cntvct.astype("float") * (10 ** 9)
    return core_power


def get_L2_power(L2_counter, freq, cntvct):
    L2_power = L2_counter.mean() * freq * 32
    L2_power /= cntvct.mean() * (10 ** 9)
    return L2_power


def get_mem_power(mem_counter, freq, cntvct):
    mem_power = mem_counter.mean() * freq * 256
    mem_power /= cntvct.mean() * (10 ** 9)
    return mem_power


def get_cmg_power(csv_dir, cmg_id, region="all"):
    csv_dir = Path(csv_dir)
    pa1 = split_csv(csv_dir.joinpath("pa1.csv"))
    pa8 = split_csv(csv_dir.joinpath("pa8.csv"))

    freq = get_freq(pa8)
    cores = get_cores(pa8, cmg_id=cmg_id)
    if cores.shape == (0,):
        return None
    cntvct = get_tag(pa1, "CNTVCT", region, cores)
    core_counter = get_tag(pa8, "0x01e0", region, cores)
    L2_counter = get_tag(pa8, "0x03e0", region, cores)
    mem_counter = get_tag(pa8, "0x03e8", region, cores)

    core_power = get_core_power(core_counter, freq, cntvct)
    L2_power = get_L2_power(L2_counter, freq, cntvct)
    mem_power = get_mem_power(mem_counter, freq, cntvct)

    # fmt: off
    return dict(core=list(core_power.array),
                L2=float(L2_power),
                mem=float(mem_power))
    # fmt: on

# TODO: better call in unambiguosly like get_watt
def get_power(csv_dir):

    # csv_dir = pathlib.Path(csv_dir) # used to postfix _reps here
    nb_cmgs = 4
    power = [{} for _ in range(nb_cmgs)]
    for cmg_id in range(nb_cmgs):
        power[cmg_id] = get_cmg_power(csv_dir, cmg_id)
    return power


def get_total_power(power):
    total = 0.0
    for cmg_power in power:
        if cmg_power is not None:
            total += sum(cmg_power["core"])
            total += cmg_power["L2"]
            total += cmg_power["mem"]
    return total
