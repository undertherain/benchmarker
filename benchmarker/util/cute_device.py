# todo use regex, load patterns from file


def get_cute_device_str(device_name):
    shorts = ["GTX 980 Ti",
              "GTX 1080 Ti",
              "P100-PCIE",
              "P100-SXM2",
              "K20Xm",
              "K40c",
              "E5-2650",
              "E5-2699",
              "i7-3820",
              "i7-3930K"]
    for token in shorts:
        if token in device_name:
            return token
    return device_name
