#!/usr/bin/env python

"""
Negative Selection Algorithm
"""

import math
import random

# Generate a random vector within the given bounds
def random_vector(minmax):
    return [minmax[i][0] + (minmax[i][1] - minmax[i][0]) * random.random() for i in range(len(minmax))]

# Calculate Euclidean distance between two vectors
def euclidean_distance(c1, c2):
    return math.sqrt(sum((c1[i] - c2[i]) ** 2.0 for i in range(len(c1))))

# Check if a vector is contained within a space
def contains(vector, space):
    for i, v in enumerate(vector):
        if v < space[i][0] or v > space[i][1]:
            return False
    return True

# Check if a vector matches any pattern in a dataset within a minimum distance
def matches(vector, dataset, min_dist):
    for pattern in dataset:
        dist = euclidean_distance(vector, pattern['vector'])
        if dist <= min_dist:
            return True
    return False

# Generate detectors
def generate_detectors(max_detectors, search_space, self_dataset, min_dist):
    detectors = []
    while len(detectors) < max_detectors:
        detector = {'vector': random_vector(search_space)}
        if not matches(detector['vector'], self_dataset, min_dist):
            if not matches(detector['vector'], detectors, 0.0):
                detectors.append(detector)
    return detectors

# Generate self dataset
def generate_self_dataset(num_records, self_space, search_space):
    self_dataset = []
    while len(self_dataset) < num_records:
        pattern = {'vector': random_vector(search_space)}
        if not matches(pattern['vector'], self_dataset, 0.0):
            if contains(pattern['vector'], self_space):
                self_dataset.append(pattern)
    return self_dataset

# Apply detectors to test inputs
def apply_detectors(detectors, bounds, self_dataset, min_dist, trials=50):
    correct = 0
    for i in range(trials):
        input_ = {'vector': random_vector(bounds)}
        actual = "N" if matches(input_['vector'], detectors, min_dist) else "S"
        expected = "S" if matches(input_['vector'], self_dataset, min_dist) else "N"
        correct += 1 if actual == expected else 0
        print(f"{i+1}/{trials}: predicted={actual}, expected={expected}")
    print(f"Done. Result: {correct}/{trials}")
    return correct

# Execute the algorithm
def execute(bounds, self_space, max_detect, max_self, min_dist):
    self_dataset = generate_self_dataset(max_self, self_space, bounds)
    print(f"Done: prepared {len(self_dataset)} self patterns.")
    detectors = generate_detectors(max_detect, bounds, self_dataset, min_dist)
    print(f"Done: prepared {len(detectors)} detectors.")
    apply_detectors(detectors, bounds, self_dataset, min_dist)
    return detectors

if __name__ == "__main__":
    # Problem configuration
    problem_size = 2
    search_space = [[0.0, 1.0] for _ in range(problem_size)]
    self_space = [[0.5, 1.0] for _ in range(problem_size)]
    max_self = 150

    # Algorithm configuration
    max_detectors = 300
    min_dist = 0.05

    # Execute the algorithm
    execute(search_space, self_space, max_detectors, max_self, min_dist)
