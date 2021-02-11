.PHONY: run

#for icpc
#source /opt/ohpc/pub/compiler/intel/compilers_and_libraries_2016.0.109/linux/bin/compilervars.sh  intel64
#icpc

CXXFLAGS=-Wall -O3  -fopenmp --std=c++14 -msse3  -ftree-vectorize -ftree-vectorizer-verbose=3 -fopt-info-vec

CFLAGS=-Wall -O3  -fopenmp  -ftree-vectorize -ftree-vectorizer-verbose=3 -fopt-info-vec
# -msse3 

CUDAFLAGS=-O3  -Wno-deprecated-gpu-targets -arch sm_50

all: blas blas_cblas

blas: blas.cpp config.h
	$(CXX) -o $@ blas.cpp $(CXXFLAGS)

# TODO: make blas library a parameter
# FCC -### -SSL2BLAMP -Kopenmp blas_cblas.cpp -o cblas

blas_cblas: blas_cblas.cpp config.h
	$(CXX) -o $@ blas_cblas.cpp $(CXXFLAGS) -lopenblas

run: blas_cblas blas 
	./blas
	./blas_cblas
