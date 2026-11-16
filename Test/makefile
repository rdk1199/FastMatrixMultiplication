SRC_DIRS := . ../Lib
SRC_FILES := $(foreach dir, $(SRC_DIRS), $(wildcard $(dir)/*.cpp))
LDFLAGS := -lpthread -lopenblas
CXXFLAGS := -g -mavx -mfma -O2 -std=c++23 -I../Lib
ASM_CXXFLAGS := -S $(CXXFLAGS)
NASMFLAGS := -f elf64

test: build/cuda_matrix_test.bin
	./build/cuda_matrix_test.bin

.PHONY: test build clean
build: build/cuda_matrix_test.bin

build/:
	mkdir -p $@

build/matrix_kernel.o: ../Lib/matrix_kernel.asm | build/
	nasm $(NASMFLAGS) $< -o $@

build/cuda_matrix_test.bin: $(SRC_FILES) build/matrix_kernel.o | build/
	g++ $(CXXFLAGS) $^ $(LDFLAGS) -o $@

matrix_test.asm: $(SRC_FILES)
	g++ $(ASM_CXXFLAGS) $^ $(LDFLAGS)

clean:
	rm -f build/*.o build/*.bin
