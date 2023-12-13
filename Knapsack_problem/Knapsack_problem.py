import random
import matplotlib.pyplot as plt

def hill_climbing_knapsack(weights, profits, capacity, iterations=100):
    # Generate a random initial solution
    n = len(weights)
    current_solution = [random.randint(0, 1) for i in range(n)]
    profit_history = []

    for i in range(iterations):
        # Calculate the weight and profit of the current solution
        current_weight = sum(w * sol for w, sol in zip(weights, current_solution))
        current_profit = sum(p * sol for p, sol in zip(profits, current_solution))

        # If the current weight is more than the capacity, we skip this iteration
        if current_weight > capacity:
            continue

        # Generate neighbor solutions
        neighbors = []
        for i in range(n):
            if current_solution[i] == 0:
                neighbor = current_solution.copy()
                neighbor[i] = 1
                neighbors.append(neighbor)

        # Evaluate all neighbors
        best_neighbor = None
        best_neighbor_profit = 0
        for neighbor in neighbors:
            weight = sum(w * sol for w, sol in zip(weights, neighbor))
            if weight <= capacity:
                profit = sum(p * sol for p, sol in zip(profits, neighbor))
                if profit > best_neighbor_profit:
                    best_neighbor_profit = profit
                    best_neighbor = neighbor

        # Check if any neighbor is better than the current solution
        if best_neighbor and best_neighbor_profit > current_profit:
            current_solution = best_neighbor
        
        profit_history.append(current_profit)

    final_weight = sum(w * sol for w, sol in zip(weights, current_solution))

    return profit_history

def initialize_population(pop_size, n):
    return [[random.randint(0, 1) for _ in range(n)] for _ in range(pop_size)]

def Fitness(solution, weights, profits, capacity):
    total_weight = sum(w * s for w, s in zip(weights, solution))
    total_profit = sum(p * s for p, s in zip(profits, solution))
    if total_weight > capacity:
        return 0  # Penalize solutions that exceed the capacity
    else:
        return total_profit

def Selection(population, fitnesses, num_parents):
    parents = []
    for _ in range(num_parents):
        max_fitness_idx = fitnesses.index(max(fitnesses))
        parents.append(population[max_fitness_idx])
        fitnesses[max_fitness_idx] = -99999999  # Mark as used
    return parents

def Crossover(parents, offspring_size):
    offspring = []
    crossover_point = offspring_size[1] // 2
    for k in range(offspring_size[0]):
        parent1_idx = k % len(parents)
        parent2_idx = (k + 1) % len(parents)
        offspring.append(parents[parent1_idx][:crossover_point] + parents[parent2_idx][crossover_point:])
    return offspring

def Mutation(offspring_crossover):
    for child in offspring_crossover:
        mutation_idx = random.randint(0, len(child) - 1)
        child[mutation_idx] = 1 - child[mutation_idx]
    return offspring_crossover

def genetic_algorithm(weights, profits, capacity, pop_size, num_generations, num_parents_mating):
    population = initialize_population(pop_size[0], pop_size[1])
    profit_history = []

    for generation in range(num_generations):
        fitnesses = [Fitness(individual, weights, profits, capacity) for individual in population]
        profit_history.append(max(fitnesses))

        parents = Selection(population, fitnesses[:], num_parents_mating)  # Use a copy of the fitnesses list
        offspring_crossover = Crossover(parents, (pop_size[0] - len(parents), pop_size[1]))
        offspring_mutation = Mutation(offspring_crossover)
        population = parents + offspring_mutation

    return profit_history


    plt.figure(figsize=(12, 6))
    plt.xlabel('Iterations')
    plt.ylabel('Profits')
    if(algo == "HC"):
        plt.plot(HC_profit_history, label='Hill_Climbing_Algorithm', color = "Blue")
    else:
        plt.plot(HC_profit_history, label='Genetic_Algorithm', color = "Red")

# Define the problem
Weights = [41, 50, 49, 59, 55, 57, 60]
Profits = [442, 525, 511, 593, 546, 564, 617]
Capacity = 170
Interation = 100

pop_size = (8, len(Weights))  # Population size and number of items
num_parents_mating = 4

HC_profit_history = hill_climbing_knapsack(Weights, Profits, Capacity, Interation)
GA_profit_history = genetic_algorithm(Weights, Profits, Capacity, pop_size, Interation, num_parents_mating)

plt.figure(figsize=(12, 6))
plt.plot(HC_profit_history, label='Hill Climbing Algorithm', color='blue')
plt.plot(GA_profit_history, label='Genetic Algorithm', color='red')
plt.xlabel('Iterations')
plt.ylabel('Profits')
plt.title('Comparison of Hill Climbing and Genetic Algorithm')
plt.legend()
plt.show()