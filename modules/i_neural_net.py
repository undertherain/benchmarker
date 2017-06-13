import importlib


class INeuralNet():
    def __init__(self, params):
        self.params = params
        self.params["batch_size"] = 32
        self.params["channels_first"] = True

    def load_data(self):
        params = self.params
        mod = importlib.import_module("modules.problems." + params["problem"] + ".data")
        get_data = getattr(mod, 'get_data')
        data = get_data(params)

        params["bytes_x_train"] = data[0].nbytes
        params["shape_x_train"] = data[0].shape
        return data
