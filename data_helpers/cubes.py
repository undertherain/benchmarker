import numpy as np

def get_cubes(dims=3,edge=8,channels=1,cnt=16):
	shape=tuple([cnt,channels]+[edge]*dims)
	X=np.zeros(shape,dtype=np.float32)
	Y=np.zeros((cnt,),dtype=np.int32)
	for i in range(cnt // 2):
		X[i*2,:]=np.ones(shape[1:])
		Y[i*2]=True
	return X,Y

def main():
	#this is for test
	x,y=get_cubes(2,1,4)
	print(x.shape)
	#print(a)

if __name__ == "__main__":
    main()