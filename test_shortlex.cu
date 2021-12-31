#include "fsa_reader.c"
#include "multiplier.cu"
#include "misc.cu"
#include "load_triangles.cu"
#include <stdio.h>
#include "shortlex_representative.cu"

int main(int argc, char* argv[]) {
  if (argc < 3) {
    exit(-1);
  }

  char* hg_filename = argv[1];
  char* small_triangles_filename = argv[2];
  HyperbolicGroup hyperbolic_group = parse_hyperbolic_group(hg_filename);
  host_hyperbolic_group = (HyperbolicGroup*) malloc(sizeof(HyperbolicGroup));
  memcpy(host_hyperbolic_group, &hyperbolic_group, sizeof(HyperbolicGroup));
  copy_hg_to_device();
  load_small_triangles(small_triangles_filename);

  int size = 32;
  int* word = (int*) malloc(sizeof(int) * size);
  int* result = (int*) malloc(sizeof(int) * size);
  int i;
  for (i=0; i<size; i++) {
    word[i] = 1;
  }
  compute_shortlex_representative(size, word, result);
  cudaDeviceSynchronize();

  exit(0);
}
