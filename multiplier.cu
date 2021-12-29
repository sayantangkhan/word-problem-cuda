#include "fsa_reader.c"
#include <stdio.h>

#define BLOCK_SIZE 1024 // Tweak later

typedef struct Slice {
  int start_index; // Inclusive
  int end_index; // Exclusive
} Slice;

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

__global__ void combine_paths(int next_num_word_blocks, Slice* next_slices, int num_word_blocks, Slice* slices, int* internal_path_matrix, int* temp_path_matrix, int num_states, int word_length) {
  int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_thread_id < next_num_word_blocks * num_states * num_states) {
    int word_block_index = global_thread_id/(num_states * num_states);
    int remainder = global_thread_id%(num_states * num_states);
    int initial_state = remainder/num_states;
    int final_state = remainder%num_states;

    if (word_block_index*2 + 1 == num_word_blocks) {
      next_slices[word_block_index] = slices[word_block_index*2];
      // Copying to temp_path_matrix
      int left = slices[word_block_index*2].start_index;
      int right = slices[word_block_index*2].end_index;
      int letter_index = ((initial_state * num_states) + final_state)*word_length + left;
      int i;
      for (i=0; i<(right-left); i++) {
	temp_path_matrix[letter_index + i] = internal_path_matrix[letter_index + i];
      }
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
      int i;
      for (i=0; i<(middle-left); i++) {
	temp_path_matrix[combined_path_letter_index + i] = internal_path_matrix[left_path_letter_index + i];
      }

      combined_path_letter_index = ((initial_state * num_states) + final_state)*word_length + middle;
      int right_path_letter_index = ((middle_state * num_states) + final_state)*word_length + middle;
      for(i=0; i<(right-middle); i++) {
	temp_path_matrix[combined_path_letter_index + i] = internal_path_matrix[right_path_letter_index + i];
      }
    } else {
      int combined_path_first_letter_index = ((initial_state * num_states) + final_state)*word_length + left;
      temp_path_matrix[combined_path_first_letter_index] = -1;
    }
  }
}

__global__ void multiply_in_kernel(int* internal_path_matrix, int num_states, int word_length, int* final_states, int num_final_states, int initial_state, int* device_result) {
  int i;
  for(i=0; i<num_final_states; i++) {
    int final_state = final_states[i];
    if (internal_path_matrix[((initial_state * num_states) + final_state)*word_length] != -1) {
      memcpy(device_result, &internal_path_matrix[((initial_state * num_states) + final_state)*word_length], sizeof(int) * word_length);
      return;
    }
  }
}

int multiply_with_generator(int word_length, int* word, int generator_to_multiply, int* result) {

  GeneralMultiplier* device_general_multiplier = &device_hyperbolic_group->general_multiplier;
  GeneralMultiplier* host_general_multiplier = &host_hyperbolic_group->general_multiplier;

  int *device_word;
  int total_num_threads;
  Slice *slices, *next_slices, *temp_slice;
  int *internal_path_matrix, *temp_path_matrix, *to_swap;
  int num_states = host_general_multiplier->num_states;

  // Padding the word by a single letter
  int padded_word_length = word_length + 1;
  int padding_symbol = host_general_multiplier->alphabet_size - 1;
  cudaMalloc(&device_word, sizeof(int) * (padded_word_length));
  cudaMemcpy(device_word, word, sizeof(int) * word_length, cudaMemcpyHostToDevice);
  cudaMemcpy(&device_word[word_length], &padding_symbol, sizeof(int), cudaMemcpyHostToDevice); // Adding the padding symbol

  // Allocating and setting memory for path matrices
  cudaMalloc(&slices, sizeof(Slice) * (padded_word_length));
  cudaMalloc(&next_slices, sizeof(Slice) * (padded_word_length));
  cudaMalloc(&internal_path_matrix, sizeof(int) * num_states * num_states * (padded_word_length));
  cudaMemset(internal_path_matrix, -1, sizeof(int) * num_states * num_states * (padded_word_length));
  cudaMalloc(&temp_path_matrix, sizeof(int) * num_states * num_states * (padded_word_length));
  cudaMemset(temp_path_matrix, -1, sizeof(int) * num_states * num_states * (padded_word_length));

  // Populating slices in parallel
  int num_blocks;
  if ((padded_word_length) % BLOCK_SIZE == 0) {
    num_blocks = (padded_word_length)/BLOCK_SIZE;
  } else {
    num_blocks = (padded_word_length)/BLOCK_SIZE + 1;
  }
  populate_slices<<<num_blocks, BLOCK_SIZE>>>(padded_word_length, device_general_multiplier, slices);

  // Computing size one paths in parallel
  if (((padded_word_length)*num_states) % BLOCK_SIZE == 0) {
    num_blocks = ((padded_word_length) * num_states)/BLOCK_SIZE;
  } else {
    num_blocks = ((padded_word_length) * num_states)/BLOCK_SIZE + 1;
  }
  compute_size_one_paths<<<num_blocks, BLOCK_SIZE>>>(padded_word_length, device_word, device_general_multiplier, slices, internal_path_matrix);

  // Combining paths to form larger paths
  int num_word_blocks = padded_word_length;
  int next_num_word_blocks = num_word_blocks;

  while (next_num_word_blocks > 1) {
    // Kernel combining paths, and creating new slices
    if (next_num_word_blocks % 2 == 0) {
      next_num_word_blocks = next_num_word_blocks/2;
    } else {
      next_num_word_blocks = next_num_word_blocks/2 + 1;
    }

    total_num_threads = next_num_word_blocks * num_states * num_states;
    if (total_num_threads % BLOCK_SIZE == 0) {
      num_blocks = total_num_threads/BLOCK_SIZE;
    } else {
      num_blocks = total_num_threads/BLOCK_SIZE + 1;
    }
    combine_paths<<<num_blocks, BLOCK_SIZE>>>(next_num_word_blocks, next_slices, num_word_blocks, slices, internal_path_matrix, temp_path_matrix, num_states, padded_word_length);

    // Swapping slices, lengths and buffers
    temp_slice = slices;
    slices = next_slices;
    next_slices = temp_slice;
    num_word_blocks = next_num_word_blocks;
    to_swap = internal_path_matrix;
    internal_path_matrix = temp_path_matrix;
    temp_path_matrix = to_swap;
  }

  // size_one_diagnostic<<<1,1>>>(internal_path_matrix);
  // cudaDeviceSynchronize();

  //
  int stateLabel = host_general_multiplier->state_labels[generator_to_multiply];
  int initial_state = host_general_multiplier->initial_state;
  int num_final_states = 0;
  int* host_final_states = (int*) malloc(sizeof(int) * num_states);
  int* device_final_states;
  cudaMalloc(&device_final_states, sizeof(int) * num_states);
  int i;
  for(i=0; i<num_states; i++) {
    if(host_general_multiplier->accepting_states[i] == stateLabel) {
      host_final_states[num_final_states] = i;
      num_final_states++;
    }
  }
  cudaMemcpy(device_final_states, host_final_states, sizeof(int) * num_final_states, cudaMemcpyHostToDevice);

  int *device_result;
  cudaMalloc(&device_result, sizeof(int) * padded_word_length);
  multiply_in_kernel<<<1,1>>>(internal_path_matrix, num_states, padded_word_length, device_final_states, num_final_states, initial_state, device_result);
  cudaMemcpy(result, device_result, sizeof(int)*padded_word_length, cudaMemcpyDeviceToHost);

  int actual_word_length = padded_word_length;
  while (result[actual_word_length - 1] == padding_symbol) {
    actual_word_length--;
  }

  return actual_word_length; // Temporary
}
