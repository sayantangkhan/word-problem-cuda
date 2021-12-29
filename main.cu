#include "fsa_reader.c"
#include "multiplier.cu"
#include "misc.cu"
#include <stdio.h>

int main(int argc, char* argv[]) {
  if (argc < 2) {
    exit(-1);
  }

  char* hg_filename = argv[1];
  HyperbolicGroup hyperbolic_group = parse_hyperbolic_group(hg_filename);
  host_hyperbolic_group = (HyperbolicGroup*) malloc(sizeof(HyperbolicGroup));
  memcpy(host_hyperbolic_group, &hyperbolic_group, sizeof(HyperbolicGroup));

  WordAcceptor word_acceptor = host_hyperbolic_group->word_acceptor;
  GeneralMultiplier general_multiplier = host_hyperbolic_group->general_multiplier;
  printf("Hyperbolicity constant = %d\n", host_hyperbolic_group->hyperbolicity_constant);
  printf("Word acceptor states = %d\n", word_acceptor.num_states);
  printf("Word acceptor initial state = %d\n", word_acceptor.initial_state);
  printf("General multiplier alphabet = %d\n", general_multiplier.alphabet_size);
  printf("General multiplier states = %d\n", general_multiplier.num_states);
  printf("General multiplier initial state = %d\n", general_multiplier.initial_state);
  printf("General multiplier state label = %d\n", general_multiplier.state_labels[8]);
  int initial_state = 1;
  int letter = 8;
  /* int width = word_acceptor.alphabet_size; */
  int width = (general_multiplier.alphabet_size) * (general_multiplier.alphabet_size);
  printf("%d\n", general_multiplier.transition_matrix[initial_state * width + letter]);
  /* printf("%d\n", general_multiplier.accepting_states[initial_state]); */
  copy_hg_to_device();
  // diagnostics<<<1,1>>>(device_general_multiplier);

  int word[10] = {0, 3, 1, 3, 5, 5, 3, 0};
  int word_to_multiply[4] = {1, 2, 2, 3};
  int* result = (int*) malloc(sizeof(int) * 14);
  int res_length = multiply_with_word(10, word, 4, word_to_multiply, result);
  cudaDeviceSynchronize();
  int i;
  for (i=0; i<res_length; i++) {
    printf("%d ", result[i]);
  }
  printf("\n");
  exit(0);
}
