#!/usr/bin/env python

"""
Dendritic Cell Algorithm
"""
import random

# Generate a random number within bounds
def rand_in_bounds(min_val, max_val):
    return min_val + ((max_val - min_val) * random.random())

# Generate a random vector within the given search space
def random_vector(search_space):
    return [rand_in_bounds(search_space[i][0], search_space[i][1]) for i in range(len(search_space))]

# Construct a pattern with class label, input, safe, and danger values
def construct_pattern(class_label, domain, p_safe, p_danger):
    set_ = domain[class_label]
    selection = random.randint(0, len(set_) - 1)
    pattern = {}
    pattern['class_label'] = class_label
    pattern['input'] = set_[selection]
    pattern['safe'] = random.random() * p_safe * 100
    pattern['danger'] = random.random() * p_danger * 100
    return pattern

# Generate a random pattern (either normal or anomaly)
def generate_pattern(domain, p_anomaly, p_normal, prob_create_anom=0.5):
    if random.random() < prob_create_anom:
        pattern = construct_pattern("Anomaly", domain, 1.0 - p_normal, p_anomaly)
        print(f">Generated Anomaly [{pattern['input']}]")
    else:
        pattern = construct_pattern("Normal", domain, p_normal, 1.0 - p_anomaly)
    return pattern

# Initialize a dendritic cell
def initialize_cell(thresh, cell=None):
    if cell is None:
        cell = {}
    cell['lifespan'] = 1000.0
    cell['k'] = 0.0
    cell['cms'] = 0.0
    cell['migration_threshold'] = rand_in_bounds(thresh[0], thresh[1])
    cell['antigen'] = {}
    return cell

# Store an antigen in the cell's memory
def store_antigen(cell, input_):
    if input_ not in cell['antigen']:
        cell['antigen'][input_] = 1
    else:
        cell['antigen'][input_] += 1

# Expose a cell to a pattern
def expose_cell(cell, cms, k, pattern, threshold):
    cell['cms'] += cms
    cell['k'] += k
    cell['lifespan'] -= cms
    store_antigen(cell, pattern['input'])
    if cell['lifespan'] <= 0:
        initialize_cell(threshold, cell)

# Check if a cell can migrate
def can_cell_migrate(cell):
    return cell['cms'] >= cell['migration_threshold'] and len(cell['antigen']) > 0

# Expose all cells to a pattern
def expose_all_cells(cells, pattern, threshold):
    migrants = []
    cms = pattern['safe'] + pattern['danger']
    k = pattern['danger'] - (pattern['safe'] * 2.0)
    for cell in cells:
        expose_cell(cell, cms, k, pattern, threshold)
        if can_cell_migrate(cell):
            migrants.append(cell)
            cell['class_label'] = "Anomaly" if cell['k'] > 0 else "Normal"
    return migrants

# Train the system
def train_system(domain, max_iter, num_cells, p_anomaly, p_normal, thresh):
    immature_cells = [initialize_cell(thresh) for _ in range(num_cells)]
    migrated = []
    for iteration in range(max_iter):
        pattern = generate_pattern(domain, p_anomaly, p_normal)
        migrants = expose_all_cells(immature_cells, pattern, thresh)
        for cell in migrants:
            immature_cells.remove(cell)
            immature_cells.append(initialize_cell(thresh))
            migrated.append(cell)
        print(f"> iter={iteration}, new={len(migrants)}, migrated={len(migrated)}")
    return migrated

# Classify a pattern based on migrated cells
def classify_pattern(migrated, pattern):
    input_ = pattern['input']
    num_cells, num_antigen = 0, 0
    for cell in migrated:
        if cell['class_label'] == "Anomaly" and input_ in cell['antigen']:
            num_cells += 1
            num_antigen += cell['antigen'][input_]
    mcav = num_cells / num_antigen if num_antigen > 0 else 0
    return "Anomaly" if mcav > 0.5 else "Normal"

# Test the system
def test_system(migrated, domain, p_anomaly, p_normal, num_trial=100):
    correct_norm = 0
    for _ in range(num_trial):
        pattern = construct_pattern("Normal", domain, p_normal, 1.0 - p_anomaly)
        class_label = classify_pattern(migrated, pattern)
        if class_label == "Normal":
            correct_norm += 1
    print(f"Finished testing Normal inputs {correct_norm}/{num_trial}")

    correct_anom = 0
    for _ in range(num_trial):
        pattern = construct_pattern("Anomaly", domain, 1.0 - p_normal, p_anomaly)
        class_label = classify_pattern(migrated, pattern)
        if class_label == "Anomaly":
            correct_anom += 1
    print(f"Finished testing Anomaly inputs {correct_anom}/{num_trial}")
    return [correct_norm, correct_anom]

# Execute the algorithm
def execute(domain, max_iter, num_cells, p_anom, p_norm, thresh):
    migrated = train_system(domain, max_iter, num_cells, p_anom, p_norm, thresh)
    test_system(migrated, domain, p_anom, p_norm)
    return migrated

if __name__ == "__main__":
    # Problem configuration
    domain = {}
    domain["Normal"] = list(range(50))
    domain["Anomaly"] = [(i + 1) * 10 for i in range(5)]
    domain["Normal"] = list(set(domain["Normal"]) - set(domain["Anomaly"]))
    p_anomaly = 0.70
    p_normal = 0.95

    # Algorithm configuration
    iterations = 100
    num_cells = 10
    thresh = [5, 15]

    # Execute the algorithm
    execute(domain, iterations, num_cells, p_anomaly, p_normal, thresh)
