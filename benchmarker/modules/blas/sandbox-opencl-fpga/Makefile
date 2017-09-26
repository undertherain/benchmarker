all:
	g++ -O2 -fPIC $(shell aocl compile-config ) -I./common/inc ./common/src/AOCLUtils/opencl.cpp matmul_fpga.cpp ./common/src/AOCLUtils/options.cpp $(shell aocl link-config ) -lelf -lrt -lpthread -o matmul_fpga.bin
	

emulator:
	aoc -march=emulator matmul_fpga.cl --board de5net_a7 -I$(ALTERAOCLSDKROOT)/include/kernel_headers
	CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 ./matmul_fpga.bin

	
hardware:
	aoc matmul_fpga.cl --board de5net_a7 -v --report -I$(ALTERAOCLSDKROOT)/include/kernel_headers
