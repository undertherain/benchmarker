import cv2
from timeit import default_timer as timer

from .i_neural_net import INeuralNet


class BackendOpenCV(INeuralNet):

    def __init__(self, params):
        super().__init__(params)
        self.params["channels_first"] = False
        self.params["nb_epoch"] = 1

    def run(self):
        params = self.params
        x_train, y_train = self.load_data()
        start = timer()
        for i in range(x_train.shape[0]):
            frame = x_train[0]
            blob = cv2.dnn.blobFromImage(frame,
                                         scalefactor=1.0,
                                         size=(300, 300),
                                         mean=(104.0, 177.0, 123.0))
            self.net.setInput(blob)
            predictions = self.net.forward()
            assert predictions is not None
        end = timer()
        params["time"] = (end - start) / self.params["nb_epoch"]
        params["time_epoch"] = (end - start) / self.params["nb_epoch"]
        params["framework_full"] = "opencv-" + cv2.__version__
        return self.params


def run(params):
    backend_cv = BackendOpenCV(params)
    return backend_cv.run()
