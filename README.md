# Word problem for hyperbolic groups in CUDA
A CUDA implementation of the NC^2-algorithm for the word problem in hyperbolic groups.
The algorithm was described in [this paper](https://doi.org/10.1145/129712.129723).

## Things to implement
- [x] General multiplier
- [x] Improved FSA parser
- [ ] Geodesic concatenator

### General todo
- [ ] Move FSAs to device constant memory
- [ ] Run benchmarks on other hyperbolic groups like the Von Dyck groups

### Multiplier todo
- [ ] Modify multiplier to return length of word
- [ ] Write variant that can multiply with words of length at most 2 delta

### Concatenator todo
- [ ] Write serial program to pre-compute small triangles
- [ ] Implement actual concatenator
- [ ] Write concatenator benchmarks
