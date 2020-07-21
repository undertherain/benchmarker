.PHONY: run

CXXFLAGS=-Wall -O3  -fopenmp --std=c++14 -msse3  -ftree-vectorize -ftree-vectorizer-verbose=3 -fopt-info-vec

CFLAGS=-Wall -O3  -fopenmp  -ftree-vectorize -ftree-vectorizer-verbose=3 -fopt-info-vec


all: main

main: main.cpp
	# TODO: make this customizable to build against different blas libraries
	$(CXX) -o $@ $^ $(CXXFLAGS) -lopenblas

run: main
	./main 128 128 128
