#include "fsa_reader.c"

__host__ void copy_hg_to_device() {
  HyperbolicGroup* temp_device_hyperbolic_group;
  cudaMalloc(&temp_device_hyperbolic_group, sizeof(HyperbolicGroup));

  // Copying the hyperbolicity constant
  cudaMemcpy(&temp_device_hyperbolic_group->hyperbolicity_constant, &host_hyperbolic_group->hyperbolicity_constant, sizeof(int), cudaMemcpyHostToDevice);

  // Copying word acceptor metadata and transition matrix
  cudaMemcpy(&temp_device_hyperbolic_group->word_acceptor.alphabet_size, &host_hyperbolic_group->word_acceptor.alphabet_size, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(&temp_device_hyperbolic_group->word_acceptor.num_states, &host_hyperbolic_group->word_acceptor.num_states, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(&temp_device_hyperbolic_group->word_acceptor.initial_state, &host_hyperbolic_group->word_acceptor.initial_state, sizeof(int), cudaMemcpyHostToDevice);

  int alphabet_size = host_hyperbolic_group->word_acceptor.alphabet_size;
  int wa_num_states = host_hyperbolic_group->word_acceptor.num_states;
  int* wa_transition_matrix;
  cudaMalloc(&wa_transition_matrix, sizeof(int) * alphabet_size * wa_num_states);
  cudaMemcpy(wa_transition_matrix, host_hyperbolic_group->word_acceptor.transition_matrix, sizeof(int) * alphabet_size * wa_num_states, cudaMemcpyHostToDevice);
  cudaMemcpy(&temp_device_hyperbolic_group->word_acceptor.transition_matrix, &wa_transition_matrix, sizeof(int*), cudaMemcpyHostToDevice);

  // Copying general multiplier to device
  cudaMemcpy(&temp_device_hyperbolic_group->general_multiplier.alphabet_size, &host_hyperbolic_group->general_multiplier.alphabet_size, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(&temp_device_hyperbolic_group->general_multiplier.num_states, &host_hyperbolic_group->general_multiplier.num_states, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(&temp_device_hyperbolic_group->general_multiplier.initial_state, &host_hyperbolic_group->general_multiplier.initial_state, sizeof(int), cudaMemcpyHostToDevice);

  int gm_alphabet_size = host_hyperbolic_group->general_multiplier.alphabet_size;
  int binary_alphabet_size = gm_alphabet_size * gm_alphabet_size;
  int gm_num_states = host_hyperbolic_group->general_multiplier.num_states;

  int *gm_state_labels, *gm_accepting_states, *gm_transition_matrix;

  cudaMalloc(&gm_state_labels, sizeof(int) * gm_alphabet_size);
  cudaMemcpy(gm_state_labels, host_hyperbolic_group->general_multiplier.state_labels, sizeof(int) * gm_alphabet_size, cudaMemcpyHostToDevice);
  cudaMemcpy(&temp_device_hyperbolic_group->general_multiplier.state_labels, &gm_state_labels, sizeof(int*), cudaMemcpyHostToDevice);

  cudaMalloc(&gm_accepting_states, sizeof(int) * gm_num_states);
  cudaMemcpy(gm_accepting_states, host_hyperbolic_group->general_multiplier.accepting_states, sizeof(int) * gm_num_states, cudaMemcpyHostToDevice);
  cudaMemcpy(&temp_device_hyperbolic_group->general_multiplier.accepting_states, &gm_accepting_states, sizeof(int*), cudaMemcpyHostToDevice);

  cudaMalloc(&gm_transition_matrix, sizeof(int) * gm_num_states * binary_alphabet_size);
  cudaMemcpy(gm_transition_matrix, host_hyperbolic_group->general_multiplier.transition_matrix, sizeof(int) * gm_num_states * binary_alphabet_size, cudaMemcpyHostToDevice);
  cudaMemcpy(&temp_device_hyperbolic_group->general_multiplier.transition_matrix, &gm_transition_matrix, sizeof(int*), cudaMemcpyHostToDevice);

  cudaMemcpyToSymbol(device_hyperbolic_group, &temp_device_hyperbolic_group, sizeof(HyperbolicGroup*));
}
