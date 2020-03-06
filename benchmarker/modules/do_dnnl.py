import os
from benchmarker.util.abstractprocess import Process

# TODO: get this from env variable, error if not set
path_benchdnn = "/home/blackbird/Projects_heavy/DL/mkl-dnn/build/tests/benchdnn"


class Benchmark():
    def __init__(self, params, remaining_args):
        self.params = params

    def run(self):
        params = self.params
        # TODO: move this to problems
        params["problem"]["cnt_filters"] = 32
        # cnt_channels = 3
        # size_image = 224
        params["problem"]["size_kernel"] = 3
        size_batch = 64
        cnt_repeats = 20
        # TODO: add strides
        spec_run = (f"mb{params['batch_size_per_device']}"
                    f"ic3"
                    f"ih224"
                    f"oc{params['problem']['cnt_filters']}"
                    f"oh224"
                    f"kh{params['problem']['size_kernel']}"
                    f"ph1n\"myconv\"")
        print(spec_run)
        command = [os.path.join(path_benchdnn, "./benchdnn"),
                   "--conv",
                   "--mode=p",
                   spec_run]
        process = Process(command=command)
        result = process.get_output()
        std_out = result["out"]
        lines = [line for line in std_out.split() if len(line) > 0]
        print(lines)
        time = lines[-1].split(":")[-1]
        seconds = float(time) / 1000
        params["time"] = seconds
