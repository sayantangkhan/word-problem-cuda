#include "fsa_reader.c"
#include "multiplier.cu"
#include "misc.cu"
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

typedef struct WordList {
  int length;
  Word* words;
} WordList;

WordList parse_multiple_words(char* filename) {
  WordList word_list;
  FILE* fp = fopen(filename, "r");
  fscanf(fp, "%d", &word_list.length);
  word_list.words = (Word *) malloc(sizeof(Word) * word_list.length);
  int j;
  for (j=0; j<word_list.length; j++) {
    Word word;
    fscanf(fp, "%d", &word.length);
    word.word = (int *) malloc(sizeof(int) * word.length);
    int i;
    for (i=0; i<word.length; i++) {
      fscanf(fp, "%d", &word.word[i]);
    }
    word_list.words[j] = word;
  }
  return word_list;
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    exit(-1);
  }

  char* hg_filename = argv[1];
  char* wordlist_filename = argv[2];

  HyperbolicGroup hyperbolic_group = parse_hyperbolic_group(hg_filename);
  host_hyperbolic_group = (HyperbolicGroup *) malloc(sizeof(HyperbolicGroup));
  memcpy(host_hyperbolic_group, &hyperbolic_group, sizeof(HyperbolicGroup));

  WordAcceptor word_acceptor = host_hyperbolic_group->word_acceptor;
  GeneralMultiplier general_multiplier = host_hyperbolic_group->general_multiplier;
  WordList word_list = parse_multiple_words(wordlist_filename);

  copy_hg_to_device();

  int j;
  for (j=0; j<word_list.length; j++) {
    Word word = word_list.words[j];

    int generator_to_multiply = 1;
    int* result = (int*) malloc(sizeof(int) * (word.length + 1));

    // Starting recording
    float elapsed_milliseconds = 0;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Actual computation
    multiply_with_generator(word.length, word.word, generator_to_multiply, result);
    cudaDeviceSynchronize();

    // Stopping recording
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_milliseconds, start, stop);

    // Printing elapsed time
    printf("%d, %f\n", word.length, elapsed_milliseconds);
  }

  exit(0);
}
