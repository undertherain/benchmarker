import re


def get_cute_device_str(device_name):
    shorts = ["(GTX 980 Ti)",
              "(GTX \d\d\d\d Ti)",
              "(RTX \d\d\d\d Ti)",
              "(RTX \d\d\d\d)",
              "(P100-PCIE)",
              "(P100-SXM2)",
              "(V100-SXM2)",
              "(K20Xm)",
              "(K40c)",
              "(ThunderX2)",
              "(Xeon)(?:\(R\) CPU )(E5-[0-9]{4} v[0-9])",
              "(Xeon)(?:\(R\)) (Gold) ([0-9]{4})",  # 'Intel(R)_Xeon(R)_Gold_6148_CPU_@_2.40GHz'
              "(Core)(?:\(TM\) )(i5-[0-9]{4}[A-Z])",
              "(?:AMD) (Ryzen [0-9] [0-9]{4}[A-Z])",
              "(?:AMD) (Ryzen [0-9] [0-9]{4})",
              "(?:AMD) (EPYC [0-9]{4})",
              "i7-3820",
              "i7-3930K"]
    for short in shorts:
        m = re.search(short, device_name)
        if m:
            found = " ".join(m.groups())
            return found
    return device_name
