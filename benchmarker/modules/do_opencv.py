import cv2
from timeit import default_timer as timer

from .i_neural_net import INeuralNet


class Benchmark(INeuralNet):

    def __init__(self, params, extra_args):
        super().__init__(params, extra_args)
        self.params["channels_first"] = False
        if self.params["mode"] != "inference":
            raise RuntimeError("opencv only supports inference")
        if self.params["batch_size"] != 1:
            raise RuntimeError("opencv only supports batch size of 1")

    def run_internal(self):
        params = self.params
        x_train, y_train = self.load_data()
        x_train = x_train.reshape((x_train.shape[0] * x_train.shape[1],) + x_train.shape[2:])
        y_train = y_train.reshape((y_train.shape[0] * y_train.shape[1],))
        start = timer()
        for i in range(x_train.shape[0]):
            frame = x_train[0]
            # print(frame.shape, frame.dtype)
            # exit(0)
            # blob = cv2.dnn.blobFromImage(frame,
            #                              scalefactor=1.0,
            #                              size=(300, 300),
            #                              mean=(104.0, 177.0, 123.0))
            blob = cv2.dnn.blobFromImage(frame)
            self.net.setInput(blob)
            predictions = self.net.forward()
            assert predictions is not None
        end = timer()
        params["time"] = (end - start) / self.params["nb_epoch"]
        params["time_epoch"] = (end - start) / self.params["nb_epoch"]
        params["framework_full"] = "opencv-" + cv2.__version__
        return self.params
