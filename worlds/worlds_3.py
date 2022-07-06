# Two normal distributions small world

# Schwefel function
#
# min f(x) = 418.9829*n + sum(-x(i)*sin(sqrt(abs(x(i)))) minimum globalne
# Global minimum
# f(x) = 0
# x(i) = 420.9687, i=1:n, -500<x(i)<500

import numpy as np
import math
import random

VECTOR_SIZE=4
SOLUTIONS_SIZE=10

def generate_candidated_solutions(dim=1, matrix_size=1):
  matrix = []
  for s in range(matrix_size):
    arr = []
    for i in range(dim): 
      arr.append(random.randrange(-499, 499))
    matrix.append(arr)
  return matrix


def do_global_perturbation(solutions):
  perturbator_vector = np.random.normal(loc = 0, scale=1, size=(SOLUTIONS_SIZE,VECTOR_SIZE))
  new_vector =  perturbator_vector + solutions
  new_vector = np.clip(new_vector,-499.0 , 499.0)
  return new_vector

def do_local_perturbation(solutions):
  perturbator_vector = np.random.normal(loc = 0, scale=0.1, size=(SOLUTIONS_SIZE,VECTOR_SIZE))
  new_vector =  perturbator_vector + solutions
  new_vector = np.clip(new_vector,-499.0 , 499.0)
  return new_vector

# Schwefel Function:
def fitness(solutions):
  acumulator = 0
  for solution in solutions:
    acumulator += -solution * math.sin(math.sqrt(abs(solution)))
  return 418.9829 * len(solutions) + acumulator

def fitness_array(solution_matrix):
  arr = []
  for solutions in solution_matrix:
    arr.append(fitness(solutions))
  return np.array(arr)

def get_best_fitness(solution_matrix):
  # Assess fitness of the solution in each row of the matrix
  new_fitness_results = fitness_array(solution_matrix)

  best_fitness = new_fitness_results.min()
  #[0][0] because where returns array
  index = np.where(new_fitness_results == best_fitness)[0][0]
  # Pick row with the best fintess result
  return best_fitness, index


if __name__ == '__main__':
  k = 1
  # Generate first solution x0
  solutions = generate_candidated_solutions(dim=VECTOR_SIZE, matrix_size=SOLUTIONS_SIZE)

  # Assess fitness of the solution
  fitness_result = get_best_fitness(solutions)
  best_fitness = fitness_result[0]
  index_fitness = fitness_result[1]

  x0 = solutions[index_fitness]

  print(f'Best first solution: {x0}')
  print(f'First fitness: {best_fitness}')

  while k < 100000:
    # Mutate last solution by adding Gaussian random variable to it
    # It will generate matrix of new solutions
    new_local_solutions = do_local_perturbation(solutions)
    new_global_solutions = do_global_perturbation(solutions)

    local_fitness = fitness_array(new_local_solutions)
    global_fitness = fitness_array(new_global_solutions)

    for i in range(len(solutions)):
      if local_fitness[i] < global_fitness[i]:
        solutions[i] = new_local_solutions[i]
      else:
        solutions[i] = new_global_solutions[i]

    # Assess fitness of the solution in each row of the matrix
    local_fitness_result = get_best_fitness(solutions)
    best_local_fintess = local_fitness_result[0]
    best_local_index = local_fitness_result[1]

    # If picked fitness result is better than the previous one, set it as the best one
    if (best_local_fintess < best_fitness):
      best_fitness = best_local_fintess
      x0 = solutions[index_fitness]
    k += 1

  print(f'Best last solution: {x0}')
  print(f'Last fitness: {best_fitness}')




