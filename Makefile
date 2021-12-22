all: release debug

release:
	nvcc -O3 -Xptxas -O3 main.cu -o build/word_problem.release

debug:
	nvcc main.cu -o build/word_problem.debug

clean:
	rm -rf build/*
