all: release debug multiplier_benchmark

release:
	nvcc -O3 -Xptxas -O3 main.cu -o build/word_problem.release

debug:
	nvcc -rdc=true -lcudadevrt -arch=sm_35 main.cu -o build/word_problem.debug

multiplier_benchmark:
	nvcc --disable-warnings -O3 -Xptxas -O3 multiplier-benchmark.cu -o build/multiplier-benchmark

clean:
	rm -rf build/*
