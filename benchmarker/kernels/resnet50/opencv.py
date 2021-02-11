import cv2


PATH_PROTO = "/mnt/kodi/blackbird/Scry/models/3rd_party/resnet50/ResNet-50-deploy.prototxt"
PATH_WEIGHTS = "/mnt/kodi/blackbird/Scry/models/3rd_party/resnet50/resnet50.caffemodel"


def get_kernel(params, unparsed_args=None):
    assert params["mode"] == "inference"
    Net = cv2.dnn.readNetFromCaffe(PATH_PROTO, PATH_WEIGHTS)
    return Net
