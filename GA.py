import numpy as np

class GeneticAlgorithm:
    def __init__(self, population_size, chromosome_length, crossover_rate, mutation_rate, num_generations, stall_generations):
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.num_generations = num_generations
        self.stall_generations = stall_generations

    def initialize_population(self):
        return np.random.randint(2, size=(self.population_size, self.chromosome_length))

    def fitness_function( chromosome):
        # Example fitness function: Count the number of ones in the chromosome
        pass
    
    def selection(self, population):
        # Roulette-wheel selection
        fitness_values = np.array([self.fitness_function(chromosome) for chromosome in population])
        cumulative_fitness = np.cumsum(fitness_values)
        selection_probabilities = fitness_values / fitness_values.sum()

        # Elitism: Select the best individual directly
        elite_index = np.argmax(fitness_values)
        selected_indices = [elite_index]

        # Select the rest based on probabilities
        selected_indices.extend(np.random.choice(self.population_size, size=self.population_size - 1, p=selection_probabilities))

        return population[selected_indices]

    def crossover(self, parents):
        children = []
        for i in range(0, len(parents), 2):
            parent1, parent2 = parents[i], parents[i + 1]
            if np.random.rand() < self.crossover_rate:
                crossover_point1 = np.random.randint(1, self.chromosome_length - 1)
                crossover_point2 = np.random.randint(crossover_point1, self.chromosome_length)
                child1 = np.concatenate((parent1[:crossover_point1], parent2[crossover_point1:crossover_point2], parent1[crossover_point2:]))
                child2 = np.concatenate((parent2[:crossover_point1], parent1[crossover_point1:crossover_point2], parent2[crossover_point2:]))
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            children.extend([child1, child2])

        return np.array(children)

    def mutation(self, population):
        for i in range(len(population)):
            if np.random.rand() < self.mutation_rate:
                mutation_point = np.random.randint(self.chromosome_length)
                population[i, mutation_point] = 1 - population[i, mutation_point]

        return population

    def run(self):
        population = self.initialize_population()

        best_fitness = None
        stall_count = 0

        for generation in range(self.num_generations):
            print("Generation", generation + 1, "of", self.num_generations)
            population = self.selection(population)
            population = self.crossover(population)
            population = self.mutation(population)

            # Check termination criteria
            current_best_fitness = max([self.fitness_function(chromosome) for chromosome in population])
            if best_fitness is None or current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                stall_count = 0
            else:
                stall_count += 1
            if stall_count >= self.stall_generations:
                break

        best_chromosome = population[np.argmax([self.fitness_function(chromosome) for chromosome in population])]
        print("Best Chromosome:", best_chromosome)
        print("Best Fitness:", best_fitness)
        return best_chromosome, best_fitness

# Example usage:
# ga = GeneticAlgorithm(population_size=30, chromosome_length=8, crossover_rate=0.8, mutation_rate=0.2, num_generations=40, stall_generations=15)
# best_chromosome, best_fitness = ga.run()
# print("Best Chromosome:", best_chromosome)
# print("Best Fitness:", best_fitness)