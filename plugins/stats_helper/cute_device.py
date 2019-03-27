# todo use regex, load patterns from file
import re


def get_cute_device_str(device_name):
    shorts = ["(GTX 980 Ti)",
              "(GTX 1080 Ti)",
              "(P100-PCIE)",
              "(P100-SXM2)",
              "(V100-SXM2)",
              "(K20Xm)",
              "(K40c)",
              "(Xeon)(?:\(R\) CPU )(E5-[0-9]{4} v[0-9])",
              "(Core)(?:\(TM\) )(i5-[0-9]{4}[A-Z])",
              "(?:AMD) (Ryzen [0-9] [0-9]{4}[A-Z])",
              "i7-3820",
              "i7-3930K"]
    for short in shorts:
        m = re.search(short, device_name)
        if m:
            found = " ".join(m.groups())
            return found
    return device_name
