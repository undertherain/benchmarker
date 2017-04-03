import sys
sys.path.append("../../../data_helpers")
from cubes import get_cubes

def get_data():
	return get_cubes(dims=3,edge=32,channels=1,cnt=512)

