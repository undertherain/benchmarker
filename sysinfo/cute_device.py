#todo use regex, load patterns from file

def get_cute_device_str(s):
	shorts= ["GTX 980 Ti",
		"P100-PCIE",
		"K20Xm",
		"K40c",		
		"E5-2650",
		"E5-2699",
		"i7-3820",
		"i7-3930K",
		]
	for i in shorts:
		if i in s: return i
	return s