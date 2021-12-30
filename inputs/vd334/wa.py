names = ["g1", "g2", "g3", "g4"]
num_states = 30
initial_state = 1

# Manually added the zero state row
transitions = [
    [0, 0, 0, 0],
    [2, 3, 4, 5],
    [0, 0, 6, 5],
    [0, 0, 6, 7],
    [8, 9, 0, 0],
    [10, 11, 0, 0],
    [12, 9, 0, 0],
    [10, 13, 0, 0],
    [0, 0, 14, 5],
    [0, 0, 6, 15],
    [0, 0, 16, 5],
    [0, 0, 6, 17],
    [0, 0, 18, 5],
    [0, 0, 19, 18],
    [0, 20, 0, 0],
    [10, 21, 0, 0],
    [18, 9, 0, 0],
    [22, 0, 0, 0],
    [0, 0, 0, 0],
    [12, 23, 0, 0],
    [0, 0, 24, 15],
    [0, 0, 25, 0],
    [0, 0, 16, 15],
    [0, 0, 6, 26],
    [27, 9, 0, 0],
    [12, 11, 0, 0],
    [28, 18, 0, 0],
    [0, 0, 0, 29],
    [0, 0, 16, 7],
    [8, 11, 0, 0],
]

if __name__ == "__main__":
    print(len(names))
    # print(" ".join(names))
    print(num_states)
    print(initial_state)
    for row in transitions:
        print(" ".join(str(i) for i in row))
