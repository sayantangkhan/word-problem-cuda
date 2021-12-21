#include "fsa_reader.c"
#include <stdio.h>

int main(int argc, char* argv[]) {
  if (argc < 3) {
    exit(-1);
  }

  char* wa_filename = argv[1];
  char* gm_filename = argv[2];
  WordAcceptor word_acceptor = parse_word_acceptor(wa_filename);
  GeneralMultiplier general_multiplier = parse_general_multiplier(gm_filename);
  printf("Word acceptor states = %d\n", word_acceptor.num_states);
  printf("Word acceptor initial state = %d\n", word_acceptor.initial_state);
  printf("General multiplier alphabet = %d\n", general_multiplier.alphabet_size);
  printf("General multiplier states = %d\n", general_multiplier.num_states);
  printf("General multiplier initial state = %d\n", general_multiplier.initial_state);
  int initial_state = 1;
  int letter = 8;
  /* int width = word_acceptor.alphabet_size; */
  int width = (general_multiplier.alphabet_size + 1) * (general_multiplier.alphabet_size + 1) - 1;
  printf("%d\n", general_multiplier.transition_matrix[initial_state * width + letter]);
  /* printf("%d\n", general_multiplier.accepting_states[initial_state]); */
  exit(0);
}
