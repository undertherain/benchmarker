
class MPI(object):
    def __init__(self, mpi_path=None, mpicc="mpicc", mpicpp="mpic++", mpirun="mpirun", hostfile=None, nprocs="2"):
        self.mpipath  = mpi_path
        self.mpicc    = mpicc
        self.mpicpp   = mpicpp
        self.mpirun   = mpirun
        self.hostfile = hostfile
        self.nprocs   = nprocs
        self.default_args = []