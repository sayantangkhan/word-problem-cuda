# Word problem for hyperbolic groups in CUDA
A CUDA implementation of the NC^2-algorithm for the word problem in hyperbolic groups.
The algorithm was described in [this paper](https://doi.org/10.1145/129712.129723).

## Things to implement
- [x] General multiplier
- [x] Improved FSA parser
- [ ] Geodesic concatenator

### General todo
- [x] Move FSAs to global memory
- [ ] Run benchmarks on other hyperbolic groups like the Von Dyck groups
- [x] Test that ShortLex is closed under inversion (experiments seem to suggest so)
- [x] Make global variables on GPU work out

### Multiplier todo
- [x] Modify multiplier to return length of word
- [x] Write variant that can multiply with words of length at most 2 delta

### Concatenator todo
- [x] Write serial program to pre-compute small triangles
- [ ] Fix the bugs in compute_initial_segments
- [ ] Implement actual concatenator
- [ ] Write concatenator benchmarks

### Optimization todo
- [ ] Move FSA to constant memory
- [ ] Use block shared memory
