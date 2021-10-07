import importlib


class IBenchmark:
    def __init__(self, params, remaining_args):
        self.params = params
        remaining_args = self.process_params(remaining_args)
        self.get_kernel(params, remaining_args)

    def measure_power_and_run(self):
        results = self.run()
        # self.post_process() # TODO
        return results

    def process_params(self, remaining_args):
        return remaining_args

    def load_data(self):
        params = self.params
        mod = importlib.import_module(
            "benchmarker.kernels." + params["problem"]["name"] + ".data"
        )
        get_data = getattr(mod, "get_data")
        data = get_data(params)
        return data

    def get_kernel(self, params, remaining_args):
        """Default function to set `self.net`.  The derived do_* classes can
        override this function if there is some framework specific
        logic involved (e.g. GPU/TPU management etc).
        """
        path_params = f"benchmarker.kernels.{params['problem']['name']}.params"
        path_kernel = (
            f"benchmarker.kernels.{params['problem']['name']}." f"{params['framework']}"
        )
        # todo(vatai): combine tflite and tensorflow
        path_kernel = path_kernel.replace("tflite", "tensorflow")
        module_kernel = importlib.import_module(path_kernel)
        try:
            module_params = importlib.import_module(path_params)
            module_params.set_extra_params(params, remaining_args)
        except ImportError:
            assert remaining_args == [], f"unexpected args: {remaining_args}"
        # TODO: rename into kernel
        self.net = module_kernel.get_kernel(self.params)

    def post_process(self):
        pass
