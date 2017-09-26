./PHONY: run

#for icpc
#source /opt/ohpc/pub/compiler/intel/compilers_and_libraries_2016.0.109/linux/bin/compilervars.sh  intel64
#icpc

CXXFLAGS=-Wall -O3  -fopenmp --std=c++14 -msse3  -ftree-vectorize -ftree-vectorizer-verbose=3 -fopt-info-vec

CFLAGS=-Wall -O3  -fopenmp -msse3  -ftree-vectorize -ftree-vectorizer-verbose=3 -fopt-info-vec

CUDAFLAGS=-O3  -Wno-deprecated-gpu-targets -arch sm_50
all: run

blas: blas.cpp
	$(CXX) -o $@ $^ $(CXXFLAGS)  -lopenblas

#cuda: cuda.cu
#	nvcc -o $@ $^ $(CUDAFLAGS)  

run: blas
#	./serial
	./blas
#	./cuda