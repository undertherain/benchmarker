import cv2
from .i_neural_net import INeuralNet


class BackendOpenCV(INeuralNet):
    def __init__(self, params):
        super().__init__(params)
        self.params["channels_first"] = True
        self.params["nb_epoch"] = 10

    def run(self):
        x_train, y_train = self.load_data()
        PATH_PROTO = "data/cv2/ssd/deploy.prototxt.txt"
        PATH_WEIGHTS = "data/cv2/ssd/res10_300x300_ssd_iter_140000.caffemodel"
        self.net = cv2.dnn.readNetFromCaffe(self.PATH_PROTO, self.PATH_WEIGHTS)
        blob = cv2.dnn.blobFromImage(frame,
                                     scalefactor=1.0,
                                     size=(300, 300),
                                     mean=(104.0, 177.0, 123.0))
        self.net.setInput(blob)
        predictions = self.net.forward()


def run(params):
    backend_cv = BackendOpenCV(params)
    return backend_cv.run()
