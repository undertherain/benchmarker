#icpc -std=c++11 -fopenmp -o main main.cpp -mkl
.PHONY: run

CXXFLAGS=-Wall -O3  -fopenmp --std=c++14 -msse3  -ftree-vectorize -ftree-vectorizer-verbose=3 -fopt-info-vec

CFLAGS=-Wall -O3  -fopenmp  -ftree-vectorize -ftree-vectorizer-verbose=3 -fopt-info-vec


all: main

main: main.cpp
	# mkl only works for intel compiler
	icpc -o $@ $^ $(CXXFLAGS) -mkl

run: main
	./main 128 128 128
