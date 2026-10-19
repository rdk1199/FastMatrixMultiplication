SRC_DIRS := . ../Lib
SRC_FILES := $(foreach dir, $(SRC_DIRS), $(wildcard $(dir)/*.cpp))
LDFLAGS := -lpthread
CXXFLAGS := -g -mavx -O0 -std=c++23 -I../Lib

test: cuda_matrix_test.bin
	./build/cuda_matrix_test.bin


cuda_matrix_test.bin: $(SRC_FILES)
	g++ $(CXXFLAGS) $^ $(LDFLAGS) -o build/$@

clean:
	rm -f build/*.o *.bin
