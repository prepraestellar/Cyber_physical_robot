import numpy as np
import random
from milestone2 import build_scenario 

def run_tests():
    robot_counts = [1, 2, 5, 10, 50, 100]
    runs_per_n = 20
    max_turns = 500

    print("| N robots | Runs | Valid pickups (mean) | Illegal pickups, distributed control | Illegal pickups, random baseline (mean) |")
    print("| :--- | :--- | :--- | :--- | :--- |")

    for n in robot_counts:
        valid_dist = []
        illegal_dist = []
        illegal_rand = []

        for seed in range(runs_per_n):
            # Test Distributed Control
            random.seed(seed)
            sim_dist = build_scenario(num_robots=n, max_turns=max_turns, search_mode="coverage")
            sim_dist.random_mode = False 
            sim_dist.run(renderer=None)
            valid_dist.append(len(sim_dist.valid_log))
            illegal_dist.append(len(sim_dist.illegal_log))

            # Test Random Baseline
            random.seed(seed)
            sim_rand = build_scenario(num_robots=n, max_turns=max_turns, search_mode="random")
            sim_rand.random_mode = True
            sim_rand.run(renderer=None)
            illegal_rand.append(len(sim_rand.illegal_log))

        print(f"| {n} | {runs_per_n} | {np.mean(valid_dist):.1f} | {np.mean(illegal_dist):.1f} | {np.mean(illegal_rand):.1f} |")

if __name__ == "__main__":
    run_tests()