SRC_DIRS := . ../Lib
SRC_FILES := $(foreach dir, $(SRC_DIRS), $(wildcard $(dir)/*.cpp))
LDFLAGS := -lpthread -lopenblas
CXXFLAGS := -g -mavx -mfma -O2 -std=c++23 -I../Lib
ASM_CXXFLAGS := -S $(CXXFLAGS)

test: cuda_matrix_test.bin
	sudo nice -n -20 ./build/cuda_matrix_test.bin


build: cuda_matrix_test.bin

cuda_matrix_test.bin: $(SRC_FILES)
	g++ $(CXXFLAGS) $^ $(LDFLAGS) -o build/$@

matrix_test.asm: $(SRC_FILES)
	g++ $(ASM_CXXFLAGS) $^ $(LDFLAGS)

clean:
	rm -f build/*.o *.bin
