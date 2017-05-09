import cool_processor




def run(params):
	params["problem"] = "bandwidth"

	# Setup problem
	mpi = OpenMPI()

	mod       = importlib.import_module("problems." + params["problem"]+".zmq")
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
