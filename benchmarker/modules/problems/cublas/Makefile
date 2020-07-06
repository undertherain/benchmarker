.PHONY: run

CXXFLAGS=-Wall -O3  -fopenmp --std=c++14 -msse3  -ftree-vectorize -ftree-vectorizer-verbose=3 -fopt-info-vec

CFLAGS=-Wall -O3  -fopenmp  -ftree-vectorize -ftree-vectorizer-verbose=3 -fopt-info-vec

CUDAFLAGS=-O3  -Wno-deprecated-gpu-targets -lcublas -arch sm_52

all: main

main: main.cu
	nvcc -o $@ $^ $(CUDAFLAGS)
#-lopenblas

run: main
	./main 128 128 128
