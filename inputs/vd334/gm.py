# States are 0-indexed, and state 0 is reject sink.
# Alphabet is 0-indexed.

from itertools import product

# Data copy pasted from sage output
names = ["g1", "g2", "g3", "g4", "_"]

state_labels = {
    1: "_",
    2: "g1",
    3: "g2",
    4: "g3",
    5: "g4",
}

accepting_states_raw = [
    [1, 1],
    [2, 1],
    [3, 2],
    [5, 3],
    [6, 3],
    [7, 1],
    [9, 2],
    [11, 1],
    [12, 4],
    [13, 5],
    [15, 5],
    [16, 1],
    [17, 4],
    [18, 1],
    [22, 1],
    [25, 1],
    [26, 1],
    [30, 1],
    [31, 1],
    [32, 1],
    [36, 1],
    [39, 1],
    [40, 1],
    [44, 1],
    [45, 1],
    [46, 1],
    [47, 3],
    [50, 2],
    [52, 1],
    [55, 1],
    [56, 1],
    [57, 5],
    [60, 4],
    [62, 1],
    [64, 1],
    [67, 1],
    [68, 1],
    [71, 1],
    [74, 1],
    [76, 1],
    [77, 1],
]

num_states = 78
initial_state = 1

transitions_raw = [
    [
        [1, 2],
        [2, 3],
        [4, 4],
        [5, 5],
        [6, 6],
        [7, 7],
        [8, 8],
        [10, 9],
        [12, 10],
        [13, 11],
        [14, 12],
        [15, 13],
        [16, 14],
        [18, 15],
        [19, 16],
        [20, 17],
        [21, 9],
        [22, 5],
        [23, 17],
        [24, 13],
    ],
    [[13, 18], [14, 12], [15, 13], [18, 15], [19, 16], [20, 17], [23, 17], [24, 13]],
    [[18, 19]],
    [[12, 20]],
    [],
    [[14, 21]],
    [[13, 18], [14, 12], [15, 13], [18, 15], [19, 22], [20, 17], [23, 17], [24, 13]],
    [[16, 23]],
    [],
    [[4, 24]],
    [[1, 25], [2, 3], [5, 5], [6, 6], [7, 26], [10, 9], [21, 9], [22, 5]],
    [[6, 27]],
    [],
    [[8, 28]],
    [[2, 29]],
    [[1, 30], [2, 3], [5, 5], [6, 6], [7, 31], [10, 9], [21, 9], [22, 5]],
    [],
    [[1, 32], [2, 3], [5, 5], [6, 6], [7, 26], [10, 9], [21, 9], [22, 5]],
    [[6, 33]],
    [[4, 34]],
    [[2, 35]],
    [[1, 30], [2, 3], [5, 5], [6, 6], [7, 36], [10, 9], [21, 9], [22, 5]],
    [[8, 37]],
    [[12, 38]],
    [[13, 39], [14, 12], [15, 13], [18, 15], [19, 16], [20, 17], [23, 17], [24, 13]],
    [[13, 18], [14, 12], [15, 13], [18, 15], [19, 40], [20, 17], [23, 17], [24, 13]],
    [[18, 41]],
    [[16, 42]],
    [[14, 43]],
    [[13, 44], [14, 12], [15, 13], [18, 15], [19, 16], [20, 17], [23, 17], [24, 13]],
    [[13, 18], [14, 12], [15, 13], [18, 15], [19, 45], [20, 17], [23, 17], [24, 13]],
    [[13, 46], [14, 17], [15, 13], [18, 13], [19, 16], [20, 17], [23, 17], [24, 13]],
    [[18, 47], [19, 48]],
    [[11, 5], [15, 9], [16, 49]],
    [[14, 50], [19, 51]],
    [[13, 52], [14, 17], [15, 13], [18, 13], [19, 46], [20, 17], [23, 17], [24, 13]],
    [[12, 53], [17, 9], [20, 5]],
    [[8, 54], [9, 5], [24, 9]],
    [[7, 55], [10, 9], [22, 5]],
    [[1, 30], [2, 3], [5, 5], [6, 6], [7, 56], [10, 9], [21, 9], [22, 5]],
    [[6, 57], [7, 58]],
    [[3, 9], [4, 59], [23, 5]],
    [[2, 60], [7, 61]],
    [[1, 46], [2, 9], [5, 5], [6, 5], [7, 26], [10, 9], [21, 9], [22, 5]],
    [[1, 62], [5, 5], [21, 9]],
    [],
    [[5, 9]],
    [[1, 27]],
    [[4, 63]],
    [[21, 5]],
    [[1, 29]],
    [[1, 32], [2, 3], [5, 5], [6, 6], [7, 64], [10, 9], [21, 9], [22, 5]],
    [[8, 65]],
    [[12, 66]],
    [[13, 67], [14, 12], [15, 13], [18, 15], [19, 40], [20, 17], [23, 17], [24, 13]],
    [[13, 68], [15, 13], [23, 17]],
    [[15, 17]],
    [[13, 19]],
    [[16, 69]],
    [[23, 13]],
    [[13, 21]],
    [[13, 44], [14, 12], [15, 13], [18, 15], [19, 40], [20, 17], [23, 17], [24, 13]],
    [[12, 70]],
    [[13, 18], [14, 12], [15, 13], [18, 15], [19, 71], [20, 17], [23, 17], [24, 13]],
    [[16, 72]],
    [[4, 73]],
    [[1, 74], [2, 3], [5, 5], [6, 6], [7, 26], [10, 9], [21, 9], [22, 5]],
    [[1, 32], [2, 3], [5, 5], [6, 6], [7, 31], [10, 9], [21, 9], [22, 5]],
    [[8, 75]],
    [[3, 13], [5, 17], [8, 4]],
    [[1, 76], [2, 9], [5, 5], [6, 5], [7, 46], [10, 9], [21, 9], [22, 5]],
    [[4, 8], [9, 17], [10, 13]],
    [[16, 10], [17, 13], [22, 17]],
    [[19, 77], [20, 17], [24, 13]],
    [[11, 17], [12, 14], [21, 13]],
    [[13, 44], [14, 12], [15, 13], [18, 15], [19, 22], [20, 17], [23, 17], [24, 13]],
    [[1, 25], [2, 3], [5, 5], [6, 6], [7, 31], [10, 9], [21, 9], [22, 5]],
]


# Code written by me, and common for all data
reversed_state_labels = [0 for _ in names]
for i in range(1, len(names) + 1):
    reversed_state_labels[names.index(state_labels[i])] = i

letters = {pair: index for (index, pair) in enumerate(product(names, names))}

transitions = []

transitions.append([0 for i in range(25)])
for index, row in enumerate(transitions_raw):
    normalized_row = []
    for letter in range(1, 25):
        found = False
        for pair in row:
            if pair[0] == letter:
                found = True
                normalized_row.append(pair[1])
        if not found:
            normalized_row.append(0)
    normalized_row.append(index + 1)
    transitions.append(normalized_row)

if __name__ == "__main__":
    # Print size of alphabet with padding symbol
    print(len(names))
    # Print alphabet
    # print(" ".join(names))
    # Print number of states
    print(num_states)
    # Print initial state
    print(initial_state)
    # Print number of accepting states
    print(len(accepting_states_raw))
    # Print mapping of labels to letters
    print(" ".join(str(i) for i in reversed_state_labels))
    # Print label of each state. A positive label is an accepting state.
    print(" ".join("{0} {1}".format(i[0], i[1]) for i in accepting_states_raw))
    # Print contents of transition matrix
    for row in transitions:
        print(" ".join(str(i) for i in row))
