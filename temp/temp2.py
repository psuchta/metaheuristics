# Schwefel function
#
# min f(x) = 418.9829*n + sum(-x(i)*sin(sqrt(abs(x(i)))) minimum globalne
# Global minimum
# f(x) = 0
# x(i) = 420.9687, i=1:n, -500<x(i)<500

import numpy as np
import math
import random
from scipy.constants import Boltzmann
from numpy import exp

VECTOR_SIZE=4
NEW_SOLUTIONS_SIZE=10

def get_randoms(dim=1):
  arr = []
  for i in range(dim): 
    arr.append(random.randrange(-499, 499))
  return arr

def do_perturbation(solutions):
  perturbator_vector = np.random.normal(loc = 0, scale=0.01, size=(NEW_SOLUTIONS_SIZE,VECTOR_SIZE))
  new_vector =  perturbator_vector + solutions
  new_vector = np.clip(new_vector,-499.0 , 499.0)
  return new_vector

# Schwefel Function:
def fitness(solutions):
  acumulator = 0
  for solution in solutions:
    acumulator += -solution * math.sin(math.sqrt(abs(solution)))
  return 418.9829 * len(solutions) + acumulator

def cool_temperature(init_temp, n):
  return init_temp/ math.log(n+1)

def fitness_array(solution_matrix):
  arr = []
  for solutions in solution_matrix:
    arr.append(fitness(solutions))
  return np.array(arr)

if __name__ == '__main__':
  k = 0.0000001
  n = 1
  init_temp = 50
  temperature = 50
  # Generate first solution x0
  x0 = get_randoms(dim=VECTOR_SIZE)
  # Assess fitness of the solution
  best_fitness = fitness(x0)

  print(f'Initial solution {x0}')
  print(f'Fitness {best_fitness}')

  while n < 100:
    temp_counter = 0
    while temp_counter < 1000:
      # Mutate last solution by adding Gaussian random variable to it
      # It will generate matrix of new solutions
      new_solutions = do_perturbation(x0)
      # Assess fitness of the solution in each row of the matrix
      new_fitness_results= fitness_array(new_solutions)
      # Pick row with the best fintess result
      best_local_fintess = new_fitness_results.min()
      index = np.where(new_fitness_results == best_local_fintess)
      new_x0 = new_solutions[index]
      # If picked fitness result is better than the previous one, set it as the best one
      if (best_local_fintess < best_fitness):
        best_fitness = best_local_fintess
        index = np.where(new_fitness_results == best_local_fintess)
        x0 = new_x0
      else:
        y = random.random()         
        if y < exp(-(best_local_fintess - best_fitness) / (k * temperature)):
          x0 = new_x0
          best_fitness = best_local_fintess
      temp_counter += 1
    temperature = cool_temperature(init_temp, n)
    n += 1

  print()
  print(f'Last solution {x0[0]}')
  print(f'With fitness {best_fitness}')
  print(f'End temperature {temperature}')


