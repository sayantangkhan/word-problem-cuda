#!/bin/bash

#SBATCH --job-name=word-problem-benchmark
#SBATCH --mail-user=saykhan@umich.edu
#SBATCH --account=lsa2
#SBATCH --partition=gpu
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:00:30
#SBATCH --output=/home/%u/wp-job-logs/%x-%j.log

echo "BLOCKSIZE = 1024"

echo "G = VD334; Running multiplier benchmark"
./build/multiplier-benchmark inputs/vd334/vd334.dat inputs/shortlex_benchmarks/vd334.data

echo "G = S2; Running multiplier benchmark"
./build/multiplier-benchmark inputs/s2/s2.data inputs/shortlex_benchmarks/s2.data

echo "G = VD334; Running shortlex benchmark"
./build/shortlex-benchmark inputs/vd334/vd334.dat inputs/shortlex_benchmarks/vd334.data

echo "G = S2; Running shortlex benchmark"
./build/shortlex-benchmark inputs/s2/s2.data inputs/shortlex_benchmarks/s2.data
