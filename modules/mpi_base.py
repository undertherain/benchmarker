
class MPI():
    def __init__(self, mpi_path=None, mpicc="mpicc", mpicpp="mpic++", mpirun="mpirun", hostfile="hostlist", nprocs="2"):
        self.mpicc    = compiler_c
        self.mpicpp   = compiler_cpp
        self.mpirun   = runner
        self.hostfile = hostfile
        self.nprocs   = nprocs