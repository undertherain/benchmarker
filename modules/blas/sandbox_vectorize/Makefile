./PHONY: run

CXXFLAGS=-Wall -O3  -fopenmp --std=c++14 -msse3  -ftree-vectorize -ftree-vectorizer-verbose=3 -fopt-info-vec
CFLAGS=-Wall -O3  -fopenmp -msse3  -ftree-vectorize -ftree-vectorizer-verbose=3 -fopt-info-vec

all: run

serial: serial.c
#	$(CXX) -o $@ $^ $(CXXFLAGS)  
	$(CC) -o $@ $^ $(CFLAGS)  
	objdump -D ./serial > serial.asm

run: serial
	./serial
