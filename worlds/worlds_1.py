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
  perturbator_vector = np.random.uniform(low = 0, high=0.5, size=(SOLUTIONS_SIZE,VECTOR_SIZE))
  # print(perturbator_vector)
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
  new_fitness_results= fitness_array(solution_matrix)

  best_fitness = new_fitness_results.min()
  index = np.where(new_fitness_results == best_fitness)
  # Pick row with the best fintess result
  return best_fitness


if __name__ == '__main__':
  k = 1
  # Generate first solution x0
  x0 = generate_candidated_solutions(dim=VECTOR_SIZE, matrix_size=SOLUTIONS_SIZE)

  # Assess fitness of the solution
  best_fitness = get_best_fitness(x0)

  print(f'Initial solution {x0}')
  print(f'Fitness {best_fitness}')

  while k < 100000:
    # Mutate last solution by adding Gaussian random variable to it
    # It will generate matrix of new solutions
    new_local_solutions = do_local_perturbation(x0)
    new_global_solutions = do_global_perturbation(x0)
    # Assess fitness of the solution in each row of the matrix
    local_best = get_best_fitness(new_local_solutions)
    global_best = get_best_fitness(new_global_solutions)

    # if(local_best < global_best):
      

    # If picked fitness result is better than the previous one, set it as the best one
    if (best_local_fintess < best_fitness):
      best_fitness = best_local_fintess
      index = np.where(new_fitness_results == best_local_fintess)
      x0 = new_solutions[index]
    k += 1

  print()
  print(f'Last solution {x0[0]}')
  print(f'With fitness {best_fitness}')


