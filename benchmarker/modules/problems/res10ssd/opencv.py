import cv2


PATH_PROTO = "/mnt/kodi/blackbird/Scry/models/3rd_party/res10_ssd/deploy.prototxt.txt"
PATH_WEIGHTS = "/mnt/kodi/blackbird/Scry/models/3rd_party/res10_ssd/res10_300x300_ssd_iter_140000.caffemodel"

Net = cv2.dnn.readNetFromCaffe(PATH_PROTO, PATH_WEIGHTS)
