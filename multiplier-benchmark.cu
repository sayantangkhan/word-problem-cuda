#include "fsa_reader.c"
#include "multiplier.cu"
#include <stdio.h>

typedef struct Word {
  int length;
  int* word;
} Word;

Word parse_word(char* filename) {
  FILE* fp = fopen(filename, "r");
  Word word;
  fscanf(fp, "%d", &word.length);
  word.word = (int *) malloc(sizeof(int) * word.length);
  int i;
  for (i=0; i<word.length; i++) {
    fscanf(fp, "%d", &word.word[i]);
  }
  return word;
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    exit(-1);
  }

  char* wa_filename = argv[1];
  char* gm_filename = argv[2];
  char* word_filename = argv[3];

  WordAcceptor word_acceptor = parse_word_acceptor(wa_filename);
  GeneralMultiplier general_multiplier = parse_general_multiplier(gm_filename);
  Word word = parse_word(word_filename);

  GeneralMultiplier* device_general_multiplier;
  cudaMalloc(&device_general_multiplier, sizeof(GeneralMultiplier));
  copy_general_multiplier(&general_multiplier, device_general_multiplier);

  int generator_to_multiply = 1;
  int* result = (int*) malloc(sizeof(int) * (word.length + 1));

  // Starting recording
  float elapsed_milliseconds = 0;
  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  // Actual computation
  multiply_with_generator(word.length, word.word, generator_to_multiply, device_general_multiplier, &general_multiplier, result);
  cudaDeviceSynchronize();

  // Stopping recording
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_milliseconds, start, stop);

  // Printing elapsed time
  printf("Elapsed milliseconds: %f\n", elapsed_milliseconds);
  // int i;
  // for (i=0; i< word.length+1; i++) {
  //   printf("%d ", result[i]);
  // }
  // printf("\n");
  exit(0);
}
