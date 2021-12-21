all: release debug

release:
	nvcc -O3 -Xptxas -O3 stencil.cu -o build/stencil_cuda.release

debug:
	nvcc -g -Xptxas -g stencil.cu -o build/stencil_cuda.debug
	g++ -g stencil.cpp -o build/stencil_cpu.debug

cpu:
	g++ stencil.cpp -o build/stencil_cpu.debug

clean:
	rm -rf build/*
