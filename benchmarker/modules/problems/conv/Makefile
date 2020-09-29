CXX := nvcc
TARGET := main
CUDNN_PATH := cudnn
LIBS := -L $(CUDNN_PATH)/lib64 -L/usr/local/lib
CXXFLAGS := -arch=sm_70 -std=c++11 -O2 -I $(CUDNN_PATH)/include

all: main

main: $(TARGET).cu
	$(CXX) $(CXXFLAGS) $(LIBS) $(TARGET).cu -o $(TARGET) \
	-lcudnn

.phony: clean

clean:
	rm $(TARGET) || echo -n ""
