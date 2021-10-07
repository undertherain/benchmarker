import os

from benchmarker.util.abstractprocess import Process

from .i_benchmark import IBenchmark


class IBinary(IBenchmark):
    def run(self):
        dirname = os.path.dirname(os.path.realpath(__file__))
        path_binary = os.path.join(
            dirname,
            "../kernels",
            self.params["problem"].get("name"),
            self.params["framework"],
            "main",
        )
        if not os.path.isfile(path_binary):
            raise (RuntimeError(f"{path_binary} not found, run make manually"))
        command = [
            path_binary,
            self.params["problem"]["precision"],
            *map(str, self.params["problem"]["size"]),
            str(self.params["nb_epoch"]),
        ]
        process = Process(command=command)
        result = process.get_output()
        std_out = result["out"]
        elapsed_time = float(std_out.strip())
        self.params["time_total"] = elapsed_time
        self.post_process()
