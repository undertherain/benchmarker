# g++ -std=c++14 -I${DNNLROOT}/include -L${DNNLROOT}/lib64 -fopenmp -o main main.cpp -ldnnl
.PHONY: run

CXX=g++

COURSE_DIR=${DNNLROOT}/include

LIB_DIR=${DNNLROOT}/lib64

CXXFLAGS=-Wall -O3  -fopenmp -std=c++11 -msse3  -ftree-vectorize -ftree-vectorizer-verbose=3 -fopt-info-vec

CFLAGS=-Wall -O3  -fopenmp  -ftree-vectorize -ftree-vectorizer-verbose=3 -fopt-info-vec

CPPFLAGS=-I$(COURSE_DIR)

LDFLAGS=-L$(LIB_DIR)

all: main

main: main.cpp
	$(CXX) -o $@ $^ $(CPPFLAGS) $(LDFLAGS) $(CXXFLAGS) -ldnnl

run: main
	./main 128 128 128
