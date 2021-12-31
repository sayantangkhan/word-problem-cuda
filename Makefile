all: debug multiplier_benchmark

debug:
	nvcc -rdc=true -lcudadevrt -arch=sm_35 test_shortlex.cu -o build/test_shortlex.debug
	nvcc -rdc=true -lcudadevrt -arch=sm_35 test_multiplier.cu -o build/test_multiplier.debug
	nvcc -rdc=true -lcudadevrt -arch=sm_35 triangle_loader.cu -o build/triangle_loader.debug

multiplier_benchmark:
	nvcc -rdc=true -lcudadevrt -arch=sm_35 --disable-warnings multiplier-benchmark.cu -o build/multiplier-benchmark

clean:
	rm -rf build/*
