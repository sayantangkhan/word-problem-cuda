# Word problem for hyperbolic groups in CUDA
A CUDA implementation of the NC^2-algorithm for the word problem in hyperbolic groups.
The algorithm was described in [this paper](https://doi.org/10.1145/129712.129723).

## Things to implement
- [x] General multiplier
- [ ] Geodesic concatenator
- [x] Improved FSA parser

## Things to do
- [x] Test multiplier on really large inputs
- [x] Figure out for what range of inputs is the runtime logarithmic
- [ ] See if moving data into block shared memory can speed things up
- [ ] Play around with block size to maximize occupancy
- [ ] Figure out memory usage of program
