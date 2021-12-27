#include <stdio.h>
#include <stdlib.h>

#ifndef FSA_READER
#define FSA_READER


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
  int* state_labels;
  int* accepting_states;
  int* transition_matrix;
} GeneralMultiplier;


WordAcceptor parse_word_acceptor(char* filename) {
  FILE* fp = fopen(filename, "r");
  WordAcceptor word_acceptor;
  fscanf(fp, "%d", &word_acceptor.alphabet_size);
  fscanf(fp, "%d", &word_acceptor.num_states);
  fscanf(fp, "%d", &word_acceptor.initial_state);
  word_acceptor.transition_matrix = (int*) malloc(sizeof(int) * word_acceptor.alphabet_size * word_acceptor.num_states);
  int i;

  for (i = 0; i < word_acceptor.alphabet_size * word_acceptor.num_states; i++) {
    fscanf(fp, "%d", &word_acceptor.transition_matrix[i]);
  }

  fclose(fp);
  return word_acceptor;
}

GeneralMultiplier parse_general_multiplier(char* filename) {
  FILE* fp = fopen(filename, "r");
  GeneralMultiplier general_multiplier;
  fscanf(fp, "%d", &general_multiplier.alphabet_size);
  fscanf(fp, "%d", &general_multiplier.num_states);
  fscanf(fp, "%d", &general_multiplier.initial_state);
  int binary_alphabet_size = (general_multiplier.alphabet_size) * (general_multiplier.alphabet_size);
  general_multiplier.state_labels = (int*) malloc(sizeof(int) * general_multiplier.alphabet_size);
  general_multiplier.accepting_states = (int*) malloc(sizeof(int) * general_multiplier.num_states);
  memset(general_multiplier.accepting_states, 0, sizeof(int) * general_multiplier.num_states);
  general_multiplier.transition_matrix = (int*) malloc(sizeof(int) * binary_alphabet_size * general_multiplier.num_states);
  int i, j;

  for (i = 0; i < general_multiplier.alphabet_size; i++) {
    fscanf(fp, "%d", &general_multiplier.state_labels[i]);
  }

  for (i = 0; i < general_multiplier.num_states; i++) {
    int state, label;
    fscanf(fp, "%d", &state);
    fscanf(fp, "%d", &label);
    general_multiplier.accepting_states[state] = label;
  }

  for (i = 0; i < general_multiplier.num_states; i++) {
    for (j = 0; j < binary_alphabet_size - 1; j++) {
      fscanf(fp, "%d", &general_multiplier.transition_matrix[i * binary_alphabet_size + j]);
    }
  }

  fclose(fp);
  return general_multiplier;
}

#endif // FSA_READER
