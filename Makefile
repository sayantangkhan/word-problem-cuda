all: debug multiplier_benchmark

debug:
	nvcc -rdc=true -lcudadevrt -arch=sm_35 main.cu -o build/word_problem.debug

multiplier_benchmark:
	nvcc -rdc=true -lcudadevrt -arch=sm_35 --disable-warnings multiplier-benchmark.cu -o build/multiplier-benchmark

clean:
	rm -rf build/*
