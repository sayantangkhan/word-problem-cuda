#include "fsa_reader.c"
#include <stdio.h>

#define BLOCK_SIZE 64 // Tweak later

typedef struct Slice {
  int start_index; // Inclusive
  int end_index; // Exclusive
} Slice;

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

// __global__ void diagnostics(GeneralMultiplier* general_multiplier) {
//   printf("General multiplier alphabet = %d\n", general_multiplier->alphabet_size);
//   printf("General multiplier states = %d\n", general_multiplier->num_states);
//   printf("General multiplier initial state = %d\n", general_multiplier->initial_state);
//   int initial_state = 1;
//   int letter = 8;
//   /* int width = word_acceptor.alphabet_size; */
//   int width = (general_multiplier->alphabet_size) * (general_multiplier->alphabet_size);
//   printf("%d\n", general_multiplier->transition_matrix[initial_state * width + letter]);
// }

__global__ void size_one_diagnostic(int* internal_path_matrix) {
  printf("SOD starting\n");
  int initial_state = 1;
  int final_state = 5;
  int word_index = 2;
  int word_length = 11;
  int path = internal_path_matrix[(initial_state * 105 + final_state)*word_length + word_index];
  printf("SOD %d\n", path);
}

__global__ void populate_slices(int word_length, GeneralMultiplier* general_multiplier, Slice* slices) {
  int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_thread_id < word_length) {
      Slice slice = {global_thread_id, global_thread_id + 1};
      slices[global_thread_id] = slice;
    }
}

__global__ void compute_size_one_paths(int word_length, int* word, GeneralMultiplier* general_multiplier, Slice* slices, int* internal_path_matrix) {
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
      int internal_path_matrix_index = ((initial_state * num_states) + final_state) * word_length + word_index;
      internal_path_matrix[internal_path_matrix_index] = j;
    }
  }
}

__global__ void combine_paths(int word_blocks, Slice* next_slices, int prev_word_blocks, Slice* slices, int* internal_path_matrix, int* temp_buffer, int num_states, int word_length) {
  int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_thread_id < word_blocks * num_states * num_states) {
    int word_block_index = global_thread_id/(num_states * num_states);
    int remainder = global_thread_id%(num_states * num_states);
    int initial_state = remainder/num_states;
    int final_state = remainder%num_states;

    if (word_block_index*2 + 1 == prev_word_blocks) {
      next_slices[word_block_index] = slices[word_block_index*2];
      // Copying to temp_buffer
      int left = slices[word_block_index*2].start_index;
      int right = slices[word_block_index*2].end_index;
      int letter_index = ((initial_state * num_states) + final_state)*word_length + left;
      memcpy(&temp_buffer[letter_index], &internal_path_matrix[letter_index], sizeof(int) * (right-left));
      return;
    }

    int left = slices[word_block_index*2].start_index;
    int middle = slices[word_block_index*2].end_index;
    int right = slices[word_block_index*2 + 1].end_index;
    Slice new_slice = {left, right};
    next_slices[word_block_index] = new_slice;

    int middle_state;
    int found = 0;
    for (middle_state=0; middle_state<num_states; middle_state++) {
      int left_path_first_letter_index = ((initial_state * num_states) + middle_state)*word_length + left;
      int left_path_first_letter = internal_path_matrix[left_path_first_letter_index];
      int right_path_first_letter_index = ((middle_state * num_states) + final_state)*word_length + middle;
      int right_path_first_letter = internal_path_matrix[right_path_first_letter_index];
      if ((left_path_first_letter != -1) && (right_path_first_letter != -1)) {
	found = 1;
	break;
      }
    }
    if (found) {
      int combined_path_letter_index = ((initial_state * num_states) + final_state)*word_length + left;
      int left_path_letter_index = ((initial_state * num_states) + middle_state)*word_length + left;
      memcpy(&temp_buffer[combined_path_letter_index], &internal_path_matrix[left_path_letter_index], sizeof(int) * (middle - left));

      combined_path_letter_index = ((initial_state * num_states) + final_state)*word_length + middle;
      int right_path_letter_index = ((middle_state * num_states) + final_state)*word_length + middle;
      memcpy(&temp_buffer[combined_path_letter_index], &internal_path_matrix[right_path_letter_index], sizeof(int)*(right - middle));
    } else {
      int combined_path_first_letter_index = ((initial_state * num_states) + final_state)*word_length + left;
      temp_buffer[combined_path_first_letter_index] = -1;
    }
  }
}

int multiply_with_generator(int word_length, int* word, int generator_to_multiply, GeneralMultiplier* device_general_multiplier, GeneralMultiplier* host_general_multiplier, int* result) {
  int* device_word;
  int total_num_threads;
  Slice *slices, *next_slices, *temp_slice;
  int *internal_path_matrix, *temp_buffer, *to_swap;
  int num_states = host_general_multiplier->num_states;

  int padding_symbol = host_general_multiplier->alphabet_size - 1;
  cudaMalloc(&device_word, sizeof(int) * (word_length + 1));
  cudaMemcpy(device_word, word, sizeof(int) * word_length, cudaMemcpyHostToDevice);
  cudaMemcpy(&device_word[word_length], &padding_symbol, sizeof(int), cudaMemcpyHostToDevice); // Adding the padding symbol

  // Allocating memory for path matrices
  cudaMalloc(&slices, sizeof(Slice) * (word_length + 1));
  cudaMalloc(&next_slices, sizeof(Slice) * (word_length + 1));
  cudaMalloc(&internal_path_matrix, sizeof(int) * num_states * num_states * (word_length + 1));
  cudaMemset(internal_path_matrix, -1, sizeof(int) * num_states * num_states * (word_length + 1));
  cudaMalloc(&temp_buffer, sizeof(int) * num_states * num_states * (word_length + 1));
  cudaMemset(temp_buffer, -1, sizeof(int) * num_states * num_states * (word_length + 1));

  int num_blocks;
  if ((word_length + 1) % BLOCK_SIZE == 0) {
    num_blocks = (word_length + 1)/BLOCK_SIZE;
  } else {
    num_blocks = (word_length + 1)/BLOCK_SIZE + 1;
  }
  populate_slices<<<num_blocks, BLOCK_SIZE>>>(word_length + 1, device_general_multiplier, slices);

  if (((word_length + 1)*num_states) % BLOCK_SIZE == 0) {
    num_blocks = ((word_length + 1) * num_states)/BLOCK_SIZE;
  } else {
    num_blocks = ((word_length + 1) * num_states)/BLOCK_SIZE + 1;
  }
  compute_size_one_paths<<<num_blocks, BLOCK_SIZE>>>(word_length + 1, device_word, device_general_multiplier, slices, internal_path_matrix);

  int prev_word_blocks = word_length + 1;
  int word_blocks = prev_word_blocks;

  while (word_blocks > 1) {
    // Kernel combining paths, and creating new slices
    if (word_blocks % 2 == 0) {
      word_blocks = word_blocks/2;
    } else {
      word_blocks = word_blocks/2 + 1;
    }

    total_num_threads = word_blocks * num_states * num_states;
    if (total_num_threads % BLOCK_SIZE == 0) {
      num_blocks = total_num_threads/BLOCK_SIZE;
    } else {
      num_blocks = total_num_threads/BLOCK_SIZE + 1;
    }
    combine_paths<<<num_blocks, BLOCK_SIZE>>>(word_blocks, next_slices, prev_word_blocks, slices, internal_path_matrix, temp_buffer, num_states, word_length + 1);

    // Swapping slices, lengths and buffers
    temp_slice = slices;
    slices = next_slices;
    next_slices = temp_slice;
    prev_word_blocks = word_blocks;
    to_swap = internal_path_matrix;
    internal_path_matrix = temp_buffer;
    temp_buffer = to_swap;
  }

  size_one_diagnostic<<<1,1>>>(internal_path_matrix);
  cudaDeviceSynchronize();

  return 0; // Temporary
}
