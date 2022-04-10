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
NEW_SOLUTIONS_SIZE=10

def get_randoms(dim=1):
  arr = []
  for i in range(dim): 
    arr.append(random.randrange(-499, 499))
  return arr

def do_perturbation(solutions, sigma):
  perturbator_vector = np.random.normal(loc = 0, scale=sigma, size=(NEW_SOLUTIONS_SIZE,VECTOR_SIZE))
  new_vector =  perturbator_vector + solutions
  gauss_overflow = np.random.normal(0,0.1,1)
  new_vector = np.clip(new_vector,-499.0 , 499.0)
  return new_vector

# Funkcja Schwefela:
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

if __name__ == '__main__':
  k = 1
  # Define standard deviation for generating guassian random variable
  sigma=0.01
  # Generate first solution x0
  x0 = get_randoms(dim=VECTOR_SIZE)
  # Assess fitness of the solution
  best_fitness = fitness(x0)
  
  print(f'Initial solution {x0}')
  print(f'Fitness {best_fitness}')

  while k < 100000:
    # Mutate last solution by adding Gaussian random variable to it
    # It will generate matrix of new solutions
    new_solutions = do_perturbation(x0, sigma)
    # Assess fitness of the solution in each row of the matrix
    new_fitness_results= fitness_array(new_solutions)
    # Pick row with the best fintess result
    best_local_fintess = new_fitness_results.min()
    # If picked fitness result is better than the previous one, set it as the best one
    if (best_local_fintess < best_fitness):
      best_fitness = best_local_fintess
      index = np.where(new_fitness_results == best_local_fintess)
      x0 = new_solutions[index]
      # If we found better solution thanm the last one, return Sigma to initial value
      # We will be searching in smaller area
      sigma = 0.01
      k=1
    else:
      # If we cant find better solution we have to increase searching area
      sigma += 0.01
    k += 1


  print()
  print(f'Last solution {x0[0]}')
  print(f'With fitness {best_fitness}')






