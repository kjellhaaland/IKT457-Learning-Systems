import random

from assignment_1.tsetlin import Tsetlin

# List with number of states
states = [1, 2, 3, 5, 10]

# Number of tsetlin automata to be created
num_la = 5

# Number of runs each tsetlin automata
runs = 100

for s in states:

    las = []
    total_results = [0, 0]

    for x in range(num_la):
        las.append(Tsetlin(s))

    print("# of states: {} ".format(s))

    for run in range(runs):
        results = [0, 0]

        for la in las:
            action = la.makeDecision()
            results[action] += 1
            total_results[action] += 1

        if results[1] <= 3:
            reward_probability = results[1] * 0.2
        else:
            reward_probability = 0.6 - (results[1] - 3) * 0.2

        for la in las:
            if random.random() < reward_probability:
                la.reward()
            else:
                la.penalize()

    total_runs = total_results[0] + total_results[1]
    average_yes = total_results[1] / total_runs
    print("{} yes and {} no".format(total_results[1], total_results[0]))
    print("{} total runs with {} yes".format(total_runs, average_yes))
