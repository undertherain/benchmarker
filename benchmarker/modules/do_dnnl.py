from benchmarker.util.abstractprocess import Process
# form a command line from parameters
# call DNNL bench
command = ["./benchdnn",
		   "--conv",
		   "--mode=p",
		   "mb64ic3ih224oc32oh224kh3ph1n\"myconv\""]
process = Process(command=["ls"])
result = process.get_output()
std_out = result["out"]
print(std_out)