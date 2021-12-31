#include "fsa_reader.c"
#include "load_triangles.cu"
#include "multiplier.cu"

#define LOCAL_BLOCK_SIZE 512

__global__ void compute_initial_segments(int word_length, int* device_word, int* device_result, int* temp_word, Slice* slices, int num_threads, int threshold) {
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
  int* thread_temp_word = &temp_word[left_index];
  int* thread_result = &device_result[left_index];

  int new_length = device_multiply_with_word(0, thread_result, thread_word_length, thread_word, thread_temp_word);

  printf("Thread id = %d: length = %d \n", global_thread_id, new_length);
  for (i=left_index; i<right_index; i++) {
    printf("Thread %d: %d\n", global_thread_id, device_result[i]);
  }

  right_index = left_index + new_length;
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
  if (num_threads % LOCAL_BLOCK_SIZE == 0) {
    num_blocks = num_threads/LOCAL_BLOCK_SIZE;
  } else {
    num_blocks = num_threads/LOCAL_BLOCK_SIZE + 1;
  }

  compute_initial_segments<<<num_blocks, LOCAL_BLOCK_SIZE>>>(word_length, device_word, device_result, temp_word, slices, num_threads, threshold);

  return 0;
}

int compute_shortlex_representative_naive(int word_length, int* word, int* result) {
  return 0;
}
