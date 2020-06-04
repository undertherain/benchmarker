from pathlib import Path

import cv2


def get_kernel(params, unparsed_args=None):
    proto = "res10_300x300_ssd_deploy.prototxt.txt"
    weights = "res10_300x300_ssd_iter_140000.caffemodel"

    BASE = Path("~/.cache/benchmarker/models").expanduser()
    PATH_PROTO = BASE.joinpath(proto)
    PATH_WEIGHTS = BASE.joinpath(weights)

    URL = "Download https://github.com/thegopieffect/computer_vision/raw/master/CAFFE_DNN/{} to {}"
    # TODO(vatai): make this automagically download!
    assert PATH_PROTO.exists(), URL.format(proto, str(BASE))
    assert PATH_WEIGHTS.exists(), URL.format(weights, str(BASE))
    assert params["mode"] == "inference"

    Net = cv2.dnn.readNetFromCaffe(str(PATH_PROTO), str(PATH_WEIGHTS))
    return Net
