CXX := nvcc
TARGET := main
CUDNN_PATH := cudnn
LIBS := -L $(CUDNN_PATH)/lib64 -L/usr/local/lib
CXXFLAGS := -arch=sm_80 -std=c++11 -O2 -I $(CUDNN_PATH)/include $(shell pkg-config --cflags opencv4)

all: main

main: $(TARGET).cu
	$(CXX) $(CXXFLAGS) $(LIBS) $(TARGET).cu -o $(TARGET) \
	-lcudnn -lopencv_imgcodecs -lopencv_imgproc -lopencv_core

.phony: clean

clean:
	rm $(TARGET) || echo -n ""
