
class MPI(object):
    def __init__(self, flags):
        # set defaults
        self.mpi_path = None
        self.mpicc = "mpicc"
        self.mpicpp = "mpic++"
        self.mpirun = "mpirun"
        self.hostfile = None
        self.nprocs = "2"

        self.default_args = []

        # set user-supplied flags
        try:
            self.nprocs = int(flags["np"])
        except:
            print("Requires flag for the number processes in the from:\n    np:<int>")
            exit(1)

        try:
            self.hostfile = flags["hostfile"]
        except:
            print("Requires flag for the hostfile in the form:\n    hostfile:<absolute_path>")
            exit(1)

        if "mpipath" in flags:
            self.mpipath = flags["mpipath"]

        if "mpicc" in flags:
            self.mpicc = flags["mpicc"]

        if "mpicpp" in flags:
            self.mpicpp = flags["mpicpp"]

        if "mpirun" in flags:
            self.mpirun = flags["mpirun"]

    @staticmethod
    def show_help():
        print("Printing MPI base help...\n")
        print("Required flags for this module are: ...")
        print("Supported flags for this module are: ...\n")
