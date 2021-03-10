import cv2
from timeit import default_timer as timer

from .i_neural_net import INeuralNet


class Benchmark(INeuralNet):

    def __init__(self, params, extra_args):
        super().__init__(params, extra_args)
        self.params["channels_first"] = False
        if self.params["mode"] != "inference":
            raise RuntimeError("opencv only supports inference")

    def run(self):
        params = self.params
        self.x_train, self.y_train = self.load_data()
        start = timer()
        for batch in self.x_train:
            # frame = batch[0]
            # print(frame.shape, frame.dtype)
            # exit(0)
            # blob = cv2.dnn.blobFromImage(frame,
            #                              scalefactor=1.0,
            #                              size=(300, 300),
            #                              mean=(104.0, 177.0, 123.0))
            blob = cv2.dnn.blobFromImages(batch)
            self.net.setInput(blob)
            predictions = self.net.forward()
            assert predictions is not None
        end = timer()
        params["time_total"] = end - start
        params["time_epoch"] = (end - start) / self.params["nb_epoch"]
        params["framework_full"] = "opencv-" + cv2.__version__
        return self.params
