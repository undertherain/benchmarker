import torch
from benchmarker.modules.problems.ssd300 import SSD300

def get_kernel(params, unparsed_args=None):
	ssd_cpu = SSD300()
	ssd_cpu = ssd_cpu.load_state_dict(torch.load("./ssd_cpu.pt"))
	return ssd_cpu
