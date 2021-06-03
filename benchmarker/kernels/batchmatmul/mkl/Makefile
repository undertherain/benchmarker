.PHONY: run

CXX=icpc

CXXFLAGS=-Wall -O3 -fopenmp -std=c++14 -msse3  -ftree-vectorize -mkl

all: main

main: main.cpp

run: main
	./main 128 128 128 1000