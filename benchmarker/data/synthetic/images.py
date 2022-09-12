import torch


class ImageGen():
    def __init__(self, params, cnt_channels=3, width=224, height=224):
        #self.cnt_classes = params.cnt_classes
        self.cnt_channels = cnt_channels
        self.width = width
        self.height = height
        self.cnt_batches = params["problem"]["cnt_batches_per_epoch"]
        self.batch_size = params["batch_size_per_device"]
        self.precision = params["problem"]["precision"]

    def __call__(self):
        if self.precision == "FP32":
            dtype = torch.float32
        elif self.precision == "FP16":
            dtype = torch.float16
        else:
            raise ValueError("unsupported float format")
        batches = []
        size = (self.batch_size, self.cnt_channels, self.width, self.height)
        for i in range(self.cnt_batches):
            inputs = torch.rand(*size,
                                dtype=dtype,
                                layout=torch.strided,
                                device=None,
                                requires_grad=False)
            batches.append({"pixel_values": inputs})
        return batches
