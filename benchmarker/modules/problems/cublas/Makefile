.PHONY: run

CXXFLAGS=-Wall -O3  -fopenmp --std=c++14 -msse3  -ftree-vectorize -ftree-vectorizer-verbose=3 -fopt-info-vec

CFLAGS=-Wall -O3  -fopenmp  -ftree-vectorize -ftree-vectorizer-verbose=3 -fopt-info-vec

CUDAFLAGS=-O3  -Wno-deprecated-gpu-targets -arch sm_50

all: blas

blas: blas.cpp config.h
	$(CXX) -o $@ blas.cpp $(CXXFLAGS) -lopenblas

run: blas 
	./blas
