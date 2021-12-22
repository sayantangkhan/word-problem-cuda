#include <stdio.h>
#include <stdlib.h>

#ifndef FSA_READER
#define FSA_READER

#define MAX_LEN 64

// Struct describing a word acceptor FSA
typedef struct WordAcceptor {
  int alphabet_size;
  int num_states;
  int initial_state;
  int* transition_matrix;
} WordAcceptor;

// Struct describing a general multiplier FSA
typedef struct GeneralMultiplier {
  int alphabet_size;
  int num_states;
  int initial_state;
  int* accepting_states;
  int* transition_matrix;
} GeneralMultiplier;


WordAcceptor parse_word_acceptor(char* filename) {
  FILE* fp = fopen(filename, "r");
  char buffer[MAX_LEN];
  WordAcceptor word_acceptor;
  fgets(buffer, MAX_LEN, fp);
  word_acceptor.alphabet_size = atoi(buffer);
  fgets(buffer, MAX_LEN, fp);
  word_acceptor.num_states = atoi(buffer);
  fgets(buffer, MAX_LEN, fp);
  word_acceptor.initial_state = atoi(buffer);
  word_acceptor.transition_matrix = (int*) malloc(sizeof(int) * word_acceptor.alphabet_size * word_acceptor.num_states);
  int i;

  for (i = 0; i < word_acceptor.alphabet_size * word_acceptor.num_states; i++) {
    fgets(buffer, MAX_LEN, fp);
    word_acceptor.transition_matrix[i] = atoi(buffer);
  }

  fclose(fp);
  return word_acceptor;
}

GeneralMultiplier parse_general_multiplier(char* filename) {
  FILE* fp = fopen(filename, "r");
  char buffer[MAX_LEN];
  GeneralMultiplier general_multiplier;
  fgets(buffer, MAX_LEN, fp);
  general_multiplier.alphabet_size = atoi(buffer) + 1;
  fgets(buffer, MAX_LEN, fp);
  general_multiplier.num_states = atoi(buffer);
  fgets(buffer, MAX_LEN, fp);
  general_multiplier.initial_state = atoi(buffer);
  int binary_alphabet_size = (general_multiplier.alphabet_size) * (general_multiplier.alphabet_size);
  general_multiplier.accepting_states = (int*) malloc(sizeof(int) * general_multiplier.num_states);
  general_multiplier.transition_matrix = (int*) malloc(sizeof(int) * binary_alphabet_size * general_multiplier.num_states);
  int i, j;

  for (i = 0; i < general_multiplier.num_states; i++) {
    fgets(buffer, MAX_LEN, fp);
    general_multiplier.accepting_states[i] = atoi(buffer);
  }

  for (i = 0; i < general_multiplier.num_states; i++) {
    for (j = 0; j < binary_alphabet_size - 1; j++) {
      fgets(buffer, MAX_LEN, fp);
      general_multiplier.transition_matrix[i * binary_alphabet_size + j] = atoi(buffer);
    }

    general_multiplier.transition_matrix[i * binary_alphabet_size + j] = i;
  }

  fclose(fp);
  return general_multiplier;
}

#endif // FSA_READER
