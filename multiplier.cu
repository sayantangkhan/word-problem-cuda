#include "fsa_reader.c"
#include <stdio.h>

#define BLOCK_SIZE 64 // Tweak later

typedef struct PathMatrix {
  int start_index; // Inclusive
  int end_index; // Exclusive
  int num_states;
  int* *paths; // num_states*num_states matrix of pointers to int arrays
} PathMatrix;

__host__ void copy_general_multiplier(GeneralMultiplier* host_general_multiplier, GeneralMultiplier* device_general_multiplier) {
  cudaMemcpy(&device_general_multiplier->alphabet_size, &host_general_multiplier->alphabet_size, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(&device_general_multiplier->num_states, &host_general_multiplier->num_states, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(&device_general_multiplier->initial_state, &host_general_multiplier->initial_state, sizeof(int), cudaMemcpyHostToDevice);

  int binary_alphabet_size = host_general_multiplier->alphabet_size * host_general_multiplier->alphabet_size;

  int *device_accepting_states, *device_transition_matrix;
  cudaMalloc(&device_accepting_states, sizeof(int) * host_general_multiplier->num_states);
  cudaMalloc(&device_transition_matrix, sizeof(int) * host_general_multiplier->num_states * binary_alphabet_size);
  cudaMemcpy(device_accepting_states, host_general_multiplier->accepting_states, sizeof(int) * host_general_multiplier->num_states, cudaMemcpyHostToDevice);
  cudaMemcpy(device_transition_matrix, host_general_multiplier->transition_matrix, sizeof(int) * host_general_multiplier->num_states * binary_alphabet_size, cudaMemcpyHostToDevice);

  cudaMemcpy(&device_general_multiplier->accepting_states, &device_accepting_states, sizeof(int*), cudaMemcpyHostToDevice);
  cudaMemcpy(&device_general_multiplier->transition_matrix, &device_transition_matrix, sizeof(int*), cudaMemcpyHostToDevice);
}

__device__ PathMatrix initialize_path_matrix(int start_index, int end_index, int num_states) {
  PathMatrix path_matrix;
  path_matrix.start_index = start_index;
  path_matrix.end_index = end_index;
  path_matrix.num_states = num_states;
  path_matrix.paths = (int**) malloc(sizeof(int*) * num_states * num_states);
  int i;
  for (i=0; i<num_states * num_states; i++) {
    path_matrix.paths[i] = NULL;
  }
  return path_matrix;
}

__global__ void diagnostics(GeneralMultiplier* general_multiplier) {
  printf("General multiplier alphabet = %d\n", general_multiplier->alphabet_size);
  printf("General multiplier states = %d\n", general_multiplier->num_states);
  printf("General multiplier initial state = %d\n", general_multiplier->initial_state);
  int initial_state = 1;
  int letter = 8;
  /* int width = word_acceptor.alphabet_size; */
  int width = (general_multiplier->alphabet_size) * (general_multiplier->alphabet_size);
  printf("%d\n", general_multiplier->transition_matrix[initial_state * width + letter]);
}

__global__ void populate_path_matrices(int word_length, GeneralMultiplier* general_multiplier, PathMatrix* path_matrices, PathMatrix* temp_buffer) {
  int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int num_states = general_multiplier->num_states;
  if (global_thread_id < word_length) {
      path_matrices[global_thread_id] = initialize_path_matrix(global_thread_id, global_thread_id + 1, num_states);
      temp_buffer[global_thread_id] = initialize_path_matrix(global_thread_id, global_thread_id + 1, num_states);
    }
}

__global__ void compute_size_one_paths(int word_length, int* word, GeneralMultiplier* general_multiplier, PathMatrix* path_matrices) {
  int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int num_states = general_multiplier->num_states;
  if (global_thread_id < word_length * num_states) {
    int initial_state = global_thread_id / word_length;
    int word_index = global_thread_id % word_length;
    int alphabet_size = general_multiplier->alphabet_size;
    int i = word[word_index];
    int j;
    for (j=0; j<alphabet_size; j++) {
      int transition_matrix_index = initial_state * (alphabet_size * alphabet_size) + (i*alphabet_size + j);
      int final_state = general_multiplier->transition_matrix[transition_matrix_index];
      int* path = (int*) malloc(sizeof(int));
      path[0] = j;
      path_matrices[word_index].start_index = word_index;
      path_matrices[word_index].end_index = word_index + 1;
      path_matrices[word_index].paths[initial_state*num_states + final_state] = path;
    }
  }
}

int multiply_with_generator(int word_length, int* word, int generator_to_multiply, GeneralMultiplier* device_general_multiplier, GeneralMultiplier* host_general_multiplier, int* result) {
  int* device_word;
  PathMatrix *path_matrices, *temp_buffer;

  int padding_symbol = host_general_multiplier->alphabet_size - 1;
  cudaMalloc(&device_word, sizeof(int) * (word_length + 1));
  cudaMemcpy(device_word, word, sizeof(int) * word_length, cudaMemcpyHostToDevice);
  cudaMemcpy(&device_word[word_length], &padding_symbol, sizeof(int), cudaMemcpyHostToDevice); // Adding the padding symbol

  // Allocating memory for path matrices
  cudaMalloc(&path_matrices, sizeof(PathMatrix) * (word_length + 1));
  cudaMalloc(&temp_buffer, sizeof(PathMatrix) * (word_length + 1));

  int num_blocks;
  if ((word_length + 1) % BLOCK_SIZE == 0) {
    num_blocks = (word_length + 1)/BLOCK_SIZE;
  } else {
    num_blocks = (word_length + 1)/BLOCK_SIZE + 1;
  }
  populate_path_matrices<<<num_blocks, BLOCK_SIZE>>>(word_length + 1, device_general_multiplier, path_matrices, temp_buffer);

  int num_states = host_general_multiplier->num_states;
  if (((word_length + 1)*num_states) % BLOCK_SIZE == 0) {
    num_blocks = ((word_length + 1) * num_states)/BLOCK_SIZE;
  } else {
    num_blocks = ((word_length + 1) * num_states)/BLOCK_SIZE + 1;
  }
  compute_size_one_paths<<<num_blocks, BLOCK_SIZE>>>(word_length + 1, device_word, device_general_multiplier, path_matrices);

  return 0; // Temporary
}
