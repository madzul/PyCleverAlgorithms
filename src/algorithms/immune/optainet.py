#!/usr/bin/env python

"""
Optimization Artificial Immune Network (opt-aiNet)
"""
import math
import random

# Objective function
def objective_function(vector):
    return sum(x ** 2.0 for x in vector)

# Generate a random vector within the given bounds
def random_vector(minmax):
    return [minmax[i][0] + (minmax[i][1] - minmax[i][0]) * random.random() for i in range(len(minmax))]

# Generate a random Gaussian value
def random_gaussian(mean=0.0, stdev=1.0):
    max_iterations = 1000  # Batas iterasi maksimum untuk mencegah infinite loop
    iterations = 0
    while True:
        u1 = 2 * random.random() - 1
        u2 = 2 * random.random() - 1
        w = u1 * u1 + u2 * u2
        if 0 < w <= 1:  # Pastikan w berada dalam rentang valid
            break
        iterations += 1
        if iterations > max_iterations:
            raise ValueError("Random Gaussian generation failed to converge.")
    w = math.sqrt((-2.0 * math.log(w)) / w)
    return mean + (u2 * w) * stdev

# Clone a parent cell
def clone(parent):
    return {'vector': list(parent['vector'])}

# Calculate mutation rate
def mutation_rate(beta, normalized_cost):
    return (1.0 / beta) * math.exp(-normalized_cost)

# Mutate a child cell
def mutate(beta, child, normalized_cost):
    alpha = mutation_rate(beta, normalized_cost)
    for i in range(len(child['vector'])):
        child['vector'][i] += alpha * random_gaussian()

# Clone and mutate a cell
def clone_cell(beta, num_clones, parent):
    clones = [clone(parent) for _ in range(num_clones)]
    for clone_ in clones:
        mutate(beta, clone_, parent['norm_cost'])
    for c in clones:
        c['cost'] = objective_function(c['vector'])
    clones.sort(key=lambda x: x['cost'])
    return clones[0]

# Calculate normalized cost for each cell in the population
def calculate_normalized_cost(pop):
    pop.sort(key=lambda x: x['cost'])
    range_ = pop[-1]['cost'] - pop[0]['cost']
    if range_ == 0.0:
        for p in pop:
            p['norm_cost'] = 1.0
    else:
        for p in pop:
            p['norm_cost'] = 1.0 - (p['cost'] / range_)

# Calculate average cost of the population
def average_cost(pop):
    return sum(p['cost'] for p in pop) / len(pop)

# Calculate Euclidean distance between two vectors
def distance(c1, c2):
    return math.sqrt(sum((c1[i] - c2[i]) ** 2.0 for i in range(len(c1))))

# Get the neighborhood of a cell
def get_neighborhood(cell, pop, aff_thresh):
    return [p for p in pop if distance(p['vector'], cell['vector']) < aff_thresh]

# Perform affinity suppression
def affinity_suppress(population, aff_thresh):
    pop = []
    for cell in population:
        neighbors = get_neighborhood(cell, population, aff_thresh)
        neighbors.sort(key=lambda x: x['cost'])
        if not neighbors or cell is neighbors[0]:
            pop.append(cell)
    return pop

# Main search algorithm
def search(search_space, max_gens, pop_size, num_clones, beta, num_rand, aff_thresh):
    pop = [{'vector': random_vector(search_space)} for _ in range(pop_size)]
    for c in pop:
        c['cost'] = objective_function(c['vector'])
    best = None
    for gen in range(max_gens):
        for c in pop:
            c['cost'] = objective_function(c['vector'])
        calculate_normalized_cost(pop)
        pop.sort(key=lambda x: x['cost'])
        if best is None or pop[0]['cost'] < best['cost']:
            best = pop[0]
        avg_cost = average_cost(pop)
        progeny = []
        while not progeny or average_cost(progeny) >= avg_cost:
            progeny = [clone_cell(beta, num_clones, pop[i]) for i in range(len(pop))]
        pop = affinity_suppress(progeny, aff_thresh)
        for _ in range(num_rand):
            pop.append({'vector': random_vector(search_space)})
        print(f" > gen {gen+1}, popSize={len(pop)}, fitness={best['cost']}")
    return best

if __name__ == "__main__":
    # Problem configuration
    problem_size = 2
    search_space = [[-5, 5] for _ in range(problem_size)]

    # Algorithm configuration
    max_gens = 150
    pop_size = 20
    num_clones = 10
    beta = 100
    num_rand = 2
    aff_thresh = (search_space[0][1] - search_space[0][0]) * 0.05

    # Execute the algorithm
    best = search(search_space, max_gens, pop_size, num_clones, beta, num_rand, aff_thresh)
    print(f"done! Solution: f={best['cost']}, s={best['vector']}")
