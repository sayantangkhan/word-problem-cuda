#include "fsa_reader.c"
#include "multiplier.cu"
#include "misc.cu"
#include "load_triangles.cu"
#include <stdio.h>

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

  printf("Number of small triangles = %ld\n", host_small_triangles->num_triangles);
  printf("Small edge max length = %d\n", host_small_triangles->small_edge_max_length);

  int num_triangles = host_small_triangles->num_triangles;
  int triangle_index = num_triangles - 1;
  int max_length = host_small_triangles->small_edge_max_length;
  int left_edge_length = host_small_triangles->edge_lengths[triangle_index*3];
  int right_edge_length = host_small_triangles->edge_lengths[triangle_index*3+1];
  int sum_edge_length = host_small_triangles->edge_lengths[triangle_index*3+2];

  int i, edge_index;
  edge_index = triangle_index * (4 * max_length);
  printf("Left edge length = %d\n", left_edge_length);
  for (i=0; i<left_edge_length; i++) {
    printf("%d ", host_small_triangles->triangles[edge_index+i]);
  }
  printf("\n");

  edge_index = triangle_index * (4 * max_length) + max_length;
  printf("Right edge length = %d\n", right_edge_length);
  for (i=0; i<right_edge_length; i++) {
    printf("%d ", host_small_triangles->triangles[edge_index+i]);
  }
  printf("\n");

  edge_index = triangle_index * (4 * max_length) + max_length + max_length;
  printf("Sum edge length = %d\n", sum_edge_length);
  for (i=0; i<sum_edge_length; i++) {
    printf("%d ", host_small_triangles->triangles[edge_index+i]);
  }
  printf("\n");

  printf("Memory usage = %ld bytes\n", sizeof(int) * 3 * num_triangles + sizeof(int) * max_length * 4 * num_triangles);

  // WordAcceptor word_acceptor = host_hyperbolic_group->word_acceptor;
  // GeneralMultiplier general_multiplier = host_hyperbolic_group->general_multiplier;
  // printf("Hyperbolicity constant = %d\n", host_hyperbolic_group->hyperbolicity_constant);
  // printf("Word acceptor states = %d\n", word_acceptor.num_states);
  // printf("Word acceptor initial state = %d\n", word_acceptor.initial_state);
  // printf("General multiplier alphabet = %d\n", general_multiplier.alphabet_size);
  // printf("General multiplier states = %d\n", general_multiplier.num_states);
  // printf("General multiplier initial state = %d\n", general_multiplier.initial_state);
  // printf("General multiplier state label = %d\n", general_multiplier.state_labels[8]);
  // int initial_state = 1;
  // int letter = 8;
  // /* int width = word_acceptor.alphabet_size; */
  // int width = (general_multiplier.alphabet_size) * (general_multiplier.alphabet_size);
  // printf("%d\n", general_multiplier.transition_matrix[initial_state * width + letter]);
  // /* printf("%d\n", general_multiplier.accepting_states[initial_state]); */
  // // diagnostics<<<1,1>>>(device_general_multiplier);

  // int word[10] = {0, 3, 1, 3, 5, 5, 3, 0};
  // int word_to_multiply[4] = {1, 2, 2, 3};
  // int* result = (int*) malloc(sizeof(int) * 14);
  // int res_length = multiply_with_word(10, word, 4, word_to_multiply, result);
  // cudaDeviceSynchronize();
  // int i;
  // for (i=0; i<res_length; i++) {
  //   printf("%d ", result[i]);
  // }
  // printf("\n");
  exit(0);
}
