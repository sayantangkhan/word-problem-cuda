#include "fsa_reader.c"
#include <stdio.h>

#ifndef MULTIPLIER
#define MULTIPLIER

#define BLOCK_SIZE 1024 // Tweak later

typedef struct Slice {
  int start_index; // Inclusive
  int end_index; // Exclusive
} Slice;

// __global__ void size_one_diagnostic(int* internal_path_matrix) {
//   printf("SOD starting\n");
//   int initial_state = 1;
//   int final_state = 5;
//   int word_index = 2;
//   int word_length = 11;
//   int path = internal_path_matrix[(initial_state * 105 + final_state)*word_length + word_index];
//   printf("SOD %d\n", path);
// }

__global__ void populate_slices(int word_length, Slice* slices) {
  int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_thread_id < word_length) {
      Slice slice = {global_thread_id, global_thread_id + 1};
      slices[global_thread_id] = slice;
    }
}

__global__ void compute_size_one_paths(int max_length, int word_length, int* word, Slice* slices, int* internal_path_matrix) {
  int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int num_states = device_hyperbolic_group->general_multiplier.num_states;
  if (global_thread_id < word_length * num_states) {
    int initial_state = global_thread_id / word_length;
    int word_index = global_thread_id % word_length;
    int alphabet_size = device_hyperbolic_group->general_multiplier.alphabet_size;
    int i = word[word_index];
    int j;
    for (j=0; j<alphabet_size; j++) {
      int transition_matrix_index = initial_state * (alphabet_size * alphabet_size) + (i*alphabet_size + j);
      int final_state = device_hyperbolic_group->general_multiplier.transition_matrix[transition_matrix_index];
      int internal_path_matrix_index = ((initial_state * num_states) + final_state) * max_length + word_index;
      internal_path_matrix[internal_path_matrix_index] = j;
    }
  }
}

__global__ void combine_paths(int next_num_word_blocks, Slice* next_slices, int num_word_blocks, Slice* slices, int* internal_path_matrix, int* temp_path_matrix, int num_states, int max_length, int word_length) {
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
      int letter_index = ((initial_state * num_states) + final_state)*max_length + left;
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
      int left_path_first_letter_index = ((initial_state * num_states) + middle_state)*max_length + left;
      int left_path_first_letter = internal_path_matrix[left_path_first_letter_index];
      int right_path_first_letter_index = ((middle_state * num_states) + final_state)*max_length + middle;
      int right_path_first_letter = internal_path_matrix[right_path_first_letter_index];
      if ((left_path_first_letter != -1) && (right_path_first_letter != -1)) {
	found = 1;
	break;
      }
    }
    if (found) {
      int combined_path_letter_index = ((initial_state * num_states) + final_state)*max_length + left;
      int left_path_letter_index = ((initial_state * num_states) + middle_state)*max_length + left;
      int i;
      for (i=0; i<(middle-left); i++) {
	temp_path_matrix[combined_path_letter_index + i] = internal_path_matrix[left_path_letter_index + i];
      }

      combined_path_letter_index = ((initial_state * num_states) + final_state)*max_length + middle;
      int right_path_letter_index = ((middle_state * num_states) + final_state)*max_length + middle;
      for(i=0; i<(right-middle); i++) {
	temp_path_matrix[combined_path_letter_index + i] = internal_path_matrix[right_path_letter_index + i];
      }
    } else {
      int combined_path_first_letter_index = ((initial_state * num_states) + final_state)*max_length + left;
      temp_path_matrix[combined_path_first_letter_index] = -1;
    }
  }
}

__global__ void multiply_in_kernel(int* internal_path_matrix, int num_states, int max_length, int word_length, int* final_states, int num_final_states, int initial_state, int* device_result) {
  int i;
  for(i=0; i<num_final_states; i++) {
    int final_state = final_states[i];
    if (internal_path_matrix[((initial_state * num_states) + final_state)*max_length] != -1) {
      memcpy(device_result, &internal_path_matrix[((initial_state * num_states) + final_state)*max_length], sizeof(int) * word_length);
      return;
    }
  }
}

__device__ int device_multiply_with_word(int left_word_length, int* left_word, int right_word_length, int* right_word, int* temp_word) {
  // This function assumes left word is always a geodesic. The right word need not be.
  int actual_word_length = 0;
  int max_word_length = left_word_length + right_word_length;

  // Dealing with the special cases of the empty left or right word
  if (right_word_length == 0) {
    return left_word_length;
  }
  if (left_word_length == 0) {
    memcpy(left_word, right_word, sizeof(int));
    left_word_length++;
    right_word = &right_word[1];
    right_word_length--;
  }
  if (right_word_length == 0) {
    return left_word_length;
  }

  Slice *slices, *next_slices, *temp_slice;
  int *internal_path_matrix, *temp_path_matrix, *to_swap;
  int num_states = device_hyperbolic_group->general_multiplier.num_states;
  int padding_symbol = device_hyperbolic_group->general_multiplier.alphabet_size - 1;

  // Allocating and setting memory for path matrices
  slices = (Slice*) malloc(sizeof(Slice) * (max_word_length));
  next_slices = (Slice*) malloc(sizeof(Slice) * (max_word_length));
  internal_path_matrix = (int*) malloc(sizeof(int) * num_states * num_states * max_word_length);
  memset(internal_path_matrix, -1, sizeof(int) * num_states * num_states * max_word_length);
  temp_path_matrix = (int*) malloc(sizeof(int) * num_states * num_states * max_word_length);
  memset(temp_path_matrix, -1, sizeof(int) * num_states * num_states * max_word_length);

  int padded_word_length = left_word_length + 1;
  int i;
  for (i=0; i<right_word_length; i++) {
    // Dealing with the case when left word cancels out to identity
    if (padded_word_length == 1) {
      padded_word_length++;
      left_word[0] = right_word[i];
      left_word[1] = padding_symbol;
      i++;
      if (i == right_word_length) {
	return 1;
      }
    }

    int generator_to_multiply = right_word[i];

    int num_blocks;
    if ((padded_word_length) % BLOCK_SIZE == 0) {
      num_blocks = (padded_word_length)/BLOCK_SIZE;
    } else {
      num_blocks = (padded_word_length)/BLOCK_SIZE + 1;
    }
    populate_slices<<<num_blocks, BLOCK_SIZE>>>(padded_word_length, slices);

    // Computing size one paths in parallel
    if (((padded_word_length)*num_states) % BLOCK_SIZE == 0) {
      num_blocks = ((padded_word_length) * num_states)/BLOCK_SIZE;
    } else {
      num_blocks = ((padded_word_length) * num_states)/BLOCK_SIZE + 1;
    }
    compute_size_one_paths<<<num_blocks, BLOCK_SIZE>>>(max_word_length, padded_word_length, left_word, slices, internal_path_matrix);

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

      int total_num_threads;
      total_num_threads = next_num_word_blocks * num_states * num_states;
      if (total_num_threads % BLOCK_SIZE == 0) {
	num_blocks = total_num_threads/BLOCK_SIZE;
      } else {
	num_blocks = total_num_threads/BLOCK_SIZE + 1;
      }
      combine_paths<<<num_blocks, BLOCK_SIZE>>>(next_num_word_blocks, next_slices, num_word_blocks, slices, internal_path_matrix, temp_path_matrix, num_states, max_word_length, padded_word_length);

      // Swapping slices, lengths and buffers
      temp_slice = slices;
      slices = next_slices;
      next_slices = temp_slice;
      num_word_blocks = next_num_word_blocks;
      to_swap = internal_path_matrix;
      internal_path_matrix = temp_path_matrix;
      temp_path_matrix = to_swap;
    }

    int stateLabel = device_hyperbolic_group->general_multiplier.state_labels[generator_to_multiply];
    int initial_state = device_hyperbolic_group->general_multiplier.initial_state;
    int num_final_states = 0;
    int* device_final_states = (int*) malloc(sizeof(int) * num_states);
    int j;
    for(j=0; j<num_states; j++) {
      if(device_hyperbolic_group->general_multiplier.accepting_states[j] == stateLabel) {
	device_final_states[num_final_states] = j;
	num_final_states++;
      }
    }

    multiply_in_kernel<<<1,1>>>(internal_path_matrix, num_states, max_word_length, padded_word_length, device_final_states, num_final_states, initial_state, temp_word);

    // Cleaning up values
    int k;
    for (k=0; k<max_word_length; k++) {
      left_word[k] = temp_word[k];
      temp_word[k] = padding_symbol;
    }

    memset(internal_path_matrix, -1, sizeof(int) * num_states * num_states * max_word_length);
    memset(temp_path_matrix, -1, sizeof(int) * num_states * num_states * max_word_length);

    actual_word_length = padded_word_length;
    while ((left_word[actual_word_length - 1] == padding_symbol) && (actual_word_length > 0)) {
      actual_word_length--;
    }

    padded_word_length = actual_word_length + 1;
  }

  return actual_word_length;
}


int multiply_with_word(int left_word_length, int* left_word, int right_word_length, int* right_word, int* result) {
  // This function assumes left word is always a geodesic
  int i;

  // Dealing with the special cases of the empty left or right word
  if (right_word_length == 0) {
    memcpy(result, left_word, sizeof(int) * left_word_length);
    return left_word_length;
  }
  if (left_word_length == 0) {
    memcpy(result, right_word, sizeof(int));
    left_word_length++;
    right_word = &right_word[1];
    right_word_length--;
  }
  if (right_word_length == 0) {
    return left_word_length;
  }


  GeneralMultiplier* host_gm = &host_hyperbolic_group->general_multiplier;

  int actual_word_length = 0;
  int padding_symbol = host_gm->alphabet_size - 1;
  int max_length = left_word_length + right_word_length;
  int *device_left_word, *device_result;

  int* temp_buffer = (int*) malloc(sizeof(int) * max_length);
  for (i=0; i<max_length; i++) {
    temp_buffer[i] = padding_symbol;
  }

  cudaMalloc(&device_left_word, sizeof(int) * max_length);
  cudaMalloc(&device_result, sizeof(int) * max_length);
  cudaMemcpy(device_left_word, temp_buffer, sizeof(int) * max_length, cudaMemcpyHostToDevice);
  cudaMemcpy(device_result, temp_buffer, sizeof(int) * max_length, cudaMemcpyHostToDevice);
  cudaMemcpy(device_left_word, left_word, sizeof(int) * left_word_length, cudaMemcpyHostToDevice);

  Slice *slices, *next_slices, *temp_slice;
  int *internal_path_matrix, *temp_path_matrix, *to_swap;
  int num_states = host_gm->num_states;

  // Allocating and setting memory for path matrices
  cudaMalloc(&slices, sizeof(Slice) * (max_length));
  cudaMalloc(&next_slices, sizeof(Slice) * (max_length));
  cudaMalloc(&internal_path_matrix, sizeof(int) * num_states * num_states * (max_length));
  cudaMemset(internal_path_matrix, -1, sizeof(int) * num_states * num_states * (max_length));
  cudaMalloc(&temp_path_matrix, sizeof(int) * num_states * num_states * (max_length));
  cudaMemset(temp_path_matrix, -1, sizeof(int) * num_states * num_states * (max_length));

  int padded_word_length = left_word_length + 1;
  for (i=0; i<right_word_length; i++) {
    if (padded_word_length == 1) {
      padded_word_length++;
      cudaMemcpy(&device_left_word[0], &right_word[i], sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(&device_left_word[1], &padding_symbol, sizeof(int), cudaMemcpyHostToDevice);
      i++;
      if (i == right_word_length) {
	result[0] = right_word[i-1];
	return 1;
      }
    }

    int generator_to_multiply = right_word[i];

    // Populate slices in parallel
    int num_blocks;
    if ((padded_word_length) % BLOCK_SIZE == 0) {
      num_blocks = (padded_word_length)/BLOCK_SIZE;
    } else {
      num_blocks = (padded_word_length)/BLOCK_SIZE + 1;
    }
    populate_slices<<<num_blocks, BLOCK_SIZE>>>(padded_word_length, slices);

    // Computing size one paths in parallel
    if (((padded_word_length)*num_states) % BLOCK_SIZE == 0) {
      num_blocks = ((padded_word_length) * num_states)/BLOCK_SIZE;
    } else {
      num_blocks = ((padded_word_length) * num_states)/BLOCK_SIZE + 1;
    }
    compute_size_one_paths<<<num_blocks, BLOCK_SIZE>>>(max_length, padded_word_length, device_left_word, slices, internal_path_matrix);

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

    int total_num_threads;
    total_num_threads = next_num_word_blocks * num_states * num_states;
    if (total_num_threads % BLOCK_SIZE == 0) {
      num_blocks = total_num_threads/BLOCK_SIZE;
    } else {
      num_blocks = total_num_threads/BLOCK_SIZE + 1;
    }
    combine_paths<<<num_blocks, BLOCK_SIZE>>>(next_num_word_blocks, next_slices, num_word_blocks, slices, internal_path_matrix, temp_path_matrix, num_states, max_length, padded_word_length);

    // Swapping slices, lengths and buffers
    temp_slice = slices;
    slices = next_slices;
    next_slices = temp_slice;
    num_word_blocks = next_num_word_blocks;
    to_swap = internal_path_matrix;
    internal_path_matrix = temp_path_matrix;
    temp_path_matrix = to_swap;
  }

  int stateLabel = host_gm->state_labels[generator_to_multiply];
  int initial_state = host_gm->initial_state;
  int num_final_states = 0;
  int* host_final_states = (int*) malloc(sizeof(int) * num_states);
  int* device_final_states;
  cudaMalloc(&device_final_states, sizeof(int) * num_states);
  int j;
  for(j=0; j<num_states; j++) {
    if(host_hyperbolic_group->general_multiplier.accepting_states[j] == stateLabel) {
      host_final_states[num_final_states] = j;
      num_final_states++;
    }
  }
  cudaMemcpy(device_final_states, host_final_states, sizeof(int) * num_final_states, cudaMemcpyHostToDevice);

  multiply_in_kernel<<<1,1>>>(internal_path_matrix, num_states, max_length, padded_word_length, device_final_states, num_final_states, initial_state, device_result);

  // Cleaning up values
  cudaMemcpy(device_left_word, temp_buffer, sizeof(int) * max_length, cudaMemcpyHostToDevice);
  cudaMemcpy(device_left_word, device_result, sizeof(int) * max_length, cudaMemcpyDeviceToDevice);
  cudaMemcpy(device_result, temp_buffer, sizeof(int) * max_length, cudaMemcpyHostToDevice);

  cudaMemset(internal_path_matrix, -1, sizeof(int) * num_states * num_states * (max_length));
  cudaMemset(temp_path_matrix, -1, sizeof(int) * num_states * num_states * (max_length));

  cudaMemcpy(result, device_left_word, sizeof(int)*padded_word_length, cudaMemcpyDeviceToHost);

  actual_word_length = padded_word_length;
  while (result[actual_word_length - 1] == padding_symbol) {
    actual_word_length--;
  }

  padded_word_length = actual_word_length + 1;
  }

  return actual_word_length;
}

int multiply_with_generator(int word_length, int* word, int generator_to_multiply, int* result) {
  return multiply_with_word(word_length, word, 1, &generator_to_multiply, result);
}

#endif // MULTIPLIER
