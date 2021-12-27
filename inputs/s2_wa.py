names = ["g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8"]
num_states = 37
initial_state = 1

# Manually added the zero state row
transitions = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [2, 3, 4, 5, 6, 7, 8, 9],
    [2, 0, 4, 5, 6, 7, 8, 9],
    [0, 3, 4, 5, 6, 7, 8, 9],
    [10, 3, 4, 0, 6, 11, 8, 9],
    [12, 13, 0, 5, 6, 7, 8, 9],
    [2, 3, 4, 14, 6, 0, 8, 9],
    [2, 3, 4, 5, 0, 7, 8, 15],
    [2, 16, 4, 5, 17, 7, 8, 0],
    [2, 3, 4, 5, 18, 19, 0, 9],
    [2, 0, 4, 5, 6, 7, 8, 20],
    [2, 3, 4, 5, 0, 7, 8, 21],
    [2, 0, 22, 5, 6, 7, 8, 9],
    [0, 3, 23, 5, 6, 7, 8, 9],
    [12, 24, 0, 5, 6, 7, 8, 9],
    [2, 3, 4, 5, 25, 19, 0, 9],
    [0, 3, 4, 26, 6, 7, 8, 9],
    [2, 3, 4, 27, 6, 0, 8, 9],
    [2, 3, 4, 14, 6, 0, 28, 9],
    [2, 3, 4, 5, 0, 7, 29, 15],
    [2, 3, 4, 14, 18, 0, 0, 9],
    [2, 3, 4, 5, 0, 19, 0, 9],
    [10, 3, 4, 0, 6, 0, 8, 9],
    [30, 3, 4, 0, 6, 11, 8, 9],
    [0, 3, 31, 5, 6, 7, 8, 9],
    [2, 3, 4, 14, 32, 0, 0, 9],
    [0, 13, 0, 5, 6, 7, 8, 9],
    [12, 0, 0, 5, 6, 7, 8, 9],
    [2, 0, 4, 5, 17, 7, 8, 0],
    [2, 16, 4, 5, 0, 7, 8, 0],
    [2, 0, 4, 5, 33, 7, 8, 0],
    [0, 3, 4, 0, 6, 11, 8, 9],
    [2, 3, 4, 14, 6, 0, 34, 9],
    [2, 3, 4, 35, 6, 0, 8, 9],
    [2, 36, 4, 5, 17, 7, 8, 0],
    [12, 26, 0, 5, 6, 7, 8, 9],
    [0, 3, 4, 0, 6, 7, 8, 9],
]

if __name__ == "__main__":
    print(len(names))
    # print(" ".join(names))
    print(num_states)
    print(initial_state)
    for row in transitions:
        print(" ".join(str(i) for i in row))
