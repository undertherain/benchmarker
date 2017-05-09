from mpi_base import MPI
import importlib

class OpenMPI(MPI):
    def __init__(self):
        super(OpenMPI, self).__init__()
        #self.default_args += " --bind-to core "

    def get_args(self):
        hosts = ""
        if self.hostfile == None:
            hosts = " -host localhost,localhost"
        else:
            hosts = " -f " + self.hostfile
        return " -np " + self.nprocs  + hosts + self.default_args 

def run(params={}):
    params["problem"] = "alltoall"

    # Setup problem
    mpi = OpenMPI()

    mod       = importlib.import_module("problems." + params["problem"]+".openmpi")
    get_model = getattr(mod, 'get_model')
    app       = get_model(params)

    # setup runner: command/script, hostfile
    output = app.execute(mpi.mpirun, mpi.get_args())

    params["time"]        = output["timing"]
    params["app_config"]  = output["config"]

    # Run ompi_info to get the version
    ompi_version = ""

    params["framework_full"] = "Open MPI" + "_" + ompi_version
    return params
