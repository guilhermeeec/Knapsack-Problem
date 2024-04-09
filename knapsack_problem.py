"""
Calculo do tempo de execucao de um problema da mochila usando PuLP
Autor: Rodrigo de Souza Couto - GTA/PEE/COPPE/UFRJ
"""

import time as tm
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as t_dist
from scipy.stats import norm
import pickle

# My module
from HeuristicSolver import HeuristicSolver

from pulp import *

#This mode saves the LP files (it only saves one, overwriting the others)
DEBUG_MODE = True

def knapsack_problem_solver(items, capacity, gap_threshold=0):
    # Creates the problem
    problem = LpProblem("Knapsack Problem", LpMaximize)

    # Creates the decision variables
    x = LpVariable.dicts("Item", [i for i, _, _ in items], 0, 1, LpBinary)

    # Adds the objective function
    problem += lpSum([value * x[item] for item, _, value in items])

    # Adds the capacity restriction function
    problem += lpSum([weight * x[item] for item, weight, _ in items]) <= capacity

    if DEBUG_MODE:
        # Write the LP file
        problem.writeLP("knapsack.lp")

    # Solves the problem
    start_time = tm.time()
    problem.solve(PULP_CBC_CMD(gapRel=gap_threshold))
    solve_time = tm.time() - start_time

    return value(problem.objective), solve_time

def heuristic_solver(items, capacity, gap_threshold=0):
    solver = HeuristicSolver(items, capacity)
    solver.solve(method="greedy")
    solver.print_status()
    return solver.get_objective(), solver.get_execution_time()

#For each run, this function generates random weights and values for the items
def generate_random_items(num_items, max_weight, max_value):
    return [(i, random.randint(1, max_weight), random.randint(1, max_value)) for i in range(num_items)]

#This function runs the problem multiple times to gather confidence intervals
def run_multiple_times_pulp(items, capacity, gap_threshold, num_runs=10):
    times = []
    objectives = []
    for _ in range(num_runs):
        objective, solve_time = knapsack_problem_solver(items, capacity, gap_threshold)
        times.append(solve_time)
        objectives.append(objective)
    return np.mean(times), np.std(times), np.mean(objectives), np.std(objectives)

#This function runs the problem multiple times to gather confidence intervals
def run_multiple_times_heuristic(items, capacity, gap_threshold, num_runs=10):
    times = []
    objectives = []
    for _ in range(num_runs):
        objective, solve_time = heuristic_solver(items, capacity, gap_threshold)
        times.append(solve_time)
        objectives.append(objective)
    return np.mean(times), np.std(times), np.mean(objectives), np.std(objectives)

#This function receives the results and plot a graph with time vs size
def plot_partial_results(filename, results, sizes, num_runs, confidence=0.95):
    plt.figure(figsize=(10, 6))
    plt.xscale('log')
    for gap, data in results.items():
        if data:
            sizes, means, stds = zip(*data)
            import sys
            #Evaluates the score to use in a confidence interval
            #If there are more than 30 samples, we use normal distribution
            #Else, we use a t-student
            if num_runs >= 30:
                score = norm.ppf((1 + confidence) / 2)
            else:
                score = t_dist.ppf((1 + confidence) / 2, num_runs - 1)
            #In this case, they are calculated as 1.96 times the standard deviation divided by the square
            #root of the number of runs (num_runs). This represents a 95% confidence interval using a normal distribution.
            plt.errorbar(sizes, means, yerr=score * np.array(stds) / np.sqrt(num_runs), marker='o', label=f'Gap {gap}')

    plt.title('Knapsack Problem: Execution time vs problem size for different gap values')
    plt.xlabel('Problem Size')
    plt.ylabel('Execution Time (s)')
    plt.grid(False)
    plt.legend()
    plt.xticks(sizes)
    plt.savefig(filename)
    plt.close()


def plot_partial_objective(filename, results, sizes, num_runs, confidence=0.95):
    plt.figure(figsize=(10, 6))
    plt.xscale('log')
    for gap, data in results.items():
        if data:
            sizes, means, stds = zip(*data)
            import sys
            #Evaluates the score to use in a confidence interval
            #If there are more than 30 samples, we use normal distribution
            #Else, we use a t-student
            if num_runs >= 30:
                score = norm.ppf((1 + confidence) / 2)
            else:
                score = t_dist.ppf((1 + confidence) / 2, num_runs - 1)
            #In this case, they are calculated as 1.96 times the standard deviation divided by the square
            #root of the number of runs (num_runs). This represents a 95% confidence interval using a normal distribution.
            plt.errorbar(sizes, means, yerr=score * np.array(stds) / np.sqrt(num_runs), marker='o', label=f'Gap {gap}')

    plt.title('Knapsack Problem: objective function vs problem size for different gap values')
    plt.xlabel('Problem Size')
    plt.ylabel('Objective function')
    plt.grid(False)
    plt.legend()
    plt.xticks(sizes)
    plt.savefig(filename)
    plt.close()


def plot_partial_objective_ratio(filename, results, sizes, num_runs, confidence=0.95):
    plt.figure(figsize=(10, 6))
    plt.xscale('log')
    for gap, data in results.items():
        if data:
            sizes, means, stds = zip(*data)
            import sys
            #Evaluates the score to use in a confidence interval
            #If there are more than 30 samples, we use normal distribution
            #Else, we use a t-student
            if num_runs >= 30:
                score = norm.ppf((1 + confidence) / 2)
            else:
                score = t_dist.ppf((1 + confidence) / 2, num_runs - 1)
            #In this case, they are calculated as 1.96 times the standard deviation divided by the square
            #root of the number of runs (num_runs). This represents a 95% confidence interval using a normal distribution.
            plt.errorbar(sizes, means, yerr=score * np.array(stds) / np.sqrt(num_runs), marker='o', label=f'Gap {gap}')

    plt.title('Knapsack Problem: objective function ratio vs problem size for different gap values')
    plt.xlabel('Problem Size')
    plt.ylabel('Objective function ratio')
    plt.grid(False)
    plt.legend()
    plt.xticks(sizes)
    plt.savefig(filename)
    plt.close()


def main():
    #Problem sizes to test: number of items
    sizes = [10, 1000, 10000, 100000, 1000000, 10000000]
    #sizes = [100, 1000]
    #Maximum item weight
    max_weight = 50
    #Maximum item value
    max_value = 100
    gap_thresholds = [0.4, 0.2, 0.1, 0.05, 0.02, 0]  # Gap values to test
    gap_thresholds = [0.4, 0.1, 0.02, 0]  # Gap values to test
    num_runs = 10  # Number of runs for each configuration

    results_pulp = {gap: [] for gap in gap_thresholds}
    results_heuristic = {gap: [] for gap in gap_thresholds}

    objective_pulp = {gap: [] for gap in gap_thresholds}
    objective_heuristic = {gap: [] for gap in gap_thresholds}
    objective_ratio= {gap: [] for gap in gap_thresholds}

    for size in sizes:
        for gap in gap_thresholds:
            items = generate_random_items(size, max_weight, max_value)
            capacity = size * max_weight / 2  # Capacidade da mochila

            mean_time, std_time, mean_objec_pulp, std_objec_pulp = run_multiple_times_pulp(items, capacity, gap, num_runs)
            results_pulp[gap].append((size, mean_time, std_time))
            objective_pulp[gap].append((size, mean_objec_pulp, std_objec_pulp))

            mean_time, std_time, mean_objec_heur, std_objec_heur = run_multiple_times_heuristic(items, capacity, gap, num_runs)
            results_heuristic[gap].append((size, mean_time, std_time))
            objective_heuristic[gap].append((size, mean_objec_heur, std_objec_heur))

            # Propagação de incertezas
            error = (mean_objec_heur/mean_objec_pulp) * np.sqrt((std_objec_pulp/mean_objec_pulp)**2+(std_objec_heur/mean_objec_heur)**2)
            objective_ratio[gap].append((size, mean_objec_heur/mean_objec_pulp, error))

            plot_partial_results('pulp_execution_time.pdf',results_pulp, sizes, num_runs)
            plot_partial_results('heuristic_execution_time.pdf',results_heuristic, sizes, num_runs)
           
            #plot_partial_objective('pulp_objective.pdf',objective_pulp, sizes, num_runs)
            #plot_partial_objective('heuristic_objective.pdf',objective_heuristic, sizes, num_runs)
            plot_partial_objective('objective_ratio.pdf',objective_ratio, sizes, num_runs)

    with open('pulp_results.pickle', 'wb') as f:
        pickle.dump(results_pulp, f)
    
    with open('pulp_objective.pickle', 'wb') as f:
        pickle.dump(objective_pulp, f)
    
    with open('heuristic_results.pickle', 'wb') as f:
        pickle.dump(results_heuristic, f)
    
    with open('heuristic_objective.pickle', 'wb') as f:
        pickle.dump(objective_heuristic, f)

if __name__ == "__main__":
    main()
