#!/bin/bash

#SBATCH --job-name=multiplier-benchmark
#SBATCH --mail-user=saykhan@umich.edu
#SBATCH --account=lsa2
#SBATCH --partition=gpu
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:00:30
#SBATCH --output=/home/%u/wp-job-logs/%x-%j.log

echo "G = S2, n = 16"
./build/multiplier-benchmark inputs/s2_wa.data inputs/s2_gm.data inputs/multiplier_benchmarks/s2/16.data
echo "G = S2, n = 70"
./build/multiplier-benchmark inputs/s2_wa.data inputs/s2_gm.data inputs/multiplier_benchmarks/s2/70.data
echo "G = S2, n = 148"
./build/multiplier-benchmark inputs/s2_wa.data inputs/s2_gm.data inputs/multiplier_benchmarks/s2/148.data
echo "G = S2, n = 1496"
./build/multiplier-benchmark inputs/s2_wa.data inputs/s2_gm.data inputs/multiplier_benchmarks/s2/1496.data
echo "G = S2, n = 3756"
./build/multiplier-benchmark inputs/s2_wa.data inputs/s2_gm.data inputs/multiplier_benchmarks/s2/3756.data
echo "G = S2, n = 7554"
./build/multiplier-benchmark inputs/s2_wa.data inputs/s2_gm.data inputs/multiplier_benchmarks/s2/7554.data
