# g++ -std=c++14 -I${DNNLROOT}/include -L${DNNLROOT}/lib64 -fopenmp -o main main.cpp -ldnnl
.PHONY: run

CXX=g++

#${DNNLROOT} is root directory of dnnl
COURSE_DIR=${DNNLROOT}/include

LIB_DIR=${DNNLROOT}/lib64

CXXFLAGS=-Wall -O3  -fopenmp -std=c++11 -msse3  -ftree-vectorize -ftree-vectorizer-verbose=3 -fopt-info-vec -ldnnl

CPPFLAGS=-I$(COURSE_DIR)

LDFLAGS=-L$(LIB_DIR)

all: main

main: main.cpp

run: main
	./main 128 128 128 1000
