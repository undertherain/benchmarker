import cv2


PATH_PROTO = "/mnt/kodi/blackbird/Scry/models/3rd_party/vgg16/VGG_ILSVRC_16_layers_deploy.prototxt"
PATH_WEIGHTS = "/mnt/kodi/blackbird/Scry/models/3rd_party/vgg16/VGG16_SalObjSub.caffemodel"

Net = cv2.dnn.readNetFromCaffe(PATH_PROTO, PATH_WEIGHTS)
