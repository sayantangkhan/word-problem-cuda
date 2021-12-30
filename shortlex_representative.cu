#include "fsa_reader.c"
#include "load_triangles.cu"
#include "multiplier.cu"

__global__ void compute_initial_segments(int word_length, int* device_word, int* device_result, int* temp_word, Slice* slices, int num_threads, int threshold, HyperbolicGroup* device_hyperbolic_group) {
  int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_thread_id >= num_threads) {
      return;
    }

  int padding_symbol = device_hyperbolic_group->general_multiplier.alphabet_size - 1;
  int left_index = threshold*global_thread_id;
  int right_index = threshold*(global_thread_id + 1);
  if (right_index > word_length) {
    right_index = word_length;
  }

  int i;
  for(i = left_index; i < right_index; i++) {
    device_result[i] = padding_symbol;
  }

  int thread_word_length = right_index - left_index;
  int* thread_word = &device_word[left_index];
  int* thread_result = &device_result[left_index];

  right_index = left_index + device_multiply_with_word(thread_result, right_index - left_index, thread_word, temp_word, device_hyperbolic_group);
}

int compute_shortlex_representative(int word_length, int* word, int* result) {
  int threshold = 3 * host_small_triangles->small_edge_max_length;

  int *device_word, *device_result, *temp_word;
  cudaMalloc(&device_word, sizeof(int)*word_length);
  cudaMalloc(&device_result, sizeof(int)*word_length);
  cudaMalloc(&temp_word, sizeof(int)*word_length);
  cudaMemcpy(device_word, word, sizeof(int)*word_length, cudaMemcpyHostToDevice);

  int num_threads;
  if (word_length % threshold == 0) {
    num_threads = word_length/threshold;
  } else {
    num_threads = word_length/threshold + 1;
  }

  Slice* slices, *next_slices;
  cudaMalloc(&slices, sizeof(Slice)*num_threads);
  cudaMalloc(&next_slices, sizeof(Slice)*num_threads);

  int num_blocks;
  if (num_threads % BLOCK_SIZE == 0) {
    num_blocks = num_threads/BLOCK_SIZE;
  } else {
    num_blocks = num_threads/BLOCK_SIZE + 1;
  }

  compute_initial_segments<<<num_blocks, BLOCK_SIZE>>>(word_length, device_word, device_result, temp_word, slices, num_threads, threshold, device_hyperbolic_group);

  return 0;
}

int compute_shortlex_representative_naive(int word_length, int* word, int* result) {
  return 0;
}
