import subprocess
import tempfile


class Process(object):
    def __init__(self, ptype="local", host="localhost", command=["hostname"]):
        self.proc = None
        self.type = ptype
        self.host = host
        self.command = command
        self.__start()

    def __start(self):
        self.out_file = tempfile.TemporaryFile()
        self.err_file = tempfile.TemporaryFile()

        runner = []
        if self.type == "local":
            pass
        if self.type == "ssh":
            runner = ["ssh", self.host, "-C"]
        if self.type == "mpi":
            runner = ["mpirun", "-host", self.host]

        self.command = runner + self.command
        self.proc = subprocess.Popen(self.command, stdout=self.out_file, stderr=self.err_file)

    def get_output(self):
        if self.proc is None:
            return None

        self.proc.wait()
        self.out_file.seek(0)
        out = self.out_file.read().decode()
        self.err_file.seek(0)
        err = self.err_file.read().decode()

        self.out_file.close()
        self.err_file.close()
        return {"returncode": self.proc.returncode, "out": out, "err": err}
