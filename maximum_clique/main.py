import numpy as np
import math
import random
import geopy.distance
import networkx as nx
from networkx.algorithms.approximation import clique
import matplotlib.pyplot as plt
import sys

max_clique_size = 2

def load_graph(file_name):
  node_count = 0
  graph = []
  with open(file_name) as f:
    for index, line in enumerate(f):
      splited_line = line.strip().split()
      if splited_line[0] == 'e':
        graph.append((int(splited_line[1]), int(splited_line[2])))
      elif splited_line[0] == 'p':
        node_count = int(splited_line[2])
  return graph, node_count

def fitness(graph, individual):
  fitness = 0.0
  print(individual)
  for i in range(len(individual)):
    if individual[i] == 1:
      for j in range(len(individual)):
        if i != j:
          if not graph.has_edge(i, j):
            fitness -= 5
            print('-5')
          else:
            fitness += 10
            print('+10')
    else:
      fitness -= 0.5
      print('-0.5')
  return fitness

def tournament_selection(population):
  new_population = []
  for j in range(2):
    random.shuffle(population)
    for i in range(0, population_size-1, 2):
        if fitness(graph, population[i]) > fitness(graph, population[i+1]):
            new_population.append(population[i])
        else:
            new_population.append(population[i+1])
  return new_population

def create_individual(graph_len):
  individual = []
  for i in range(graph_len):
      individual.append(random.randint(0, 1))
  return individual

def crossover(parent1, parent2, length):
  position = random.randint(2, length-2)
  child1 = parent1[0:position] + parent2[position:length]
  child2 = parent2[0:position] + parent1[position:length]
  return child1, child2

def mutation(individual,probability = 0.4):
  length = len(individual)
  check = random.uniform(0, 1)
  if(check <= probability):
      position = random.randint(0, length-1)
      # Perhaps change value in the position to the opposite number
      individual[position] = random.randint(0, 1)
  return individual

if __name__ == '__main__':
  graph = nx.Graph()

  # graph, node_count = load_graph('low.txt')
  graph_edges, node_count = load_graph('low.txt')
  graph_nodes = list(range(1, node_count + 1))
  graph.add_nodes_from(graph_nodes)
  graph.add_edges_from(graph_edges)
  # nx.draw(graph)
  # plt.show()
  # print(list(nx.find_cliques(graph, [2])))
  # sys.exit()

  generation = 0 
  population_size = 25
  population = []
  graph_len = len(graph)

  for i in range(population_size):
    individual = create_individual(graph_len)
    population.append(individual)


  best_fitness = -999999999999
  fittest_individual = None
  while(generation != 50):
    generation += 1
    population = tournament_selection(population)
    random.shuffle(population)
    new_population = []
    for i in range(0, population_size-1, 2):
        child1, child2 = crossover(population[i], population[i+1], graph_len)
        new_population.append(child1)
        new_population.append(child2)

    # for individual in new_population:
    #   if(generation < 200):
    #       individual = mutation(individual)
    #   else:
    #       individual = mutation(individual, 0.2)

    population = new_population
    for individual in population:
      local_fitness = fitness(graph, individual)
      if(local_fitness > best_fitness):
          best_fitness = local_fitness
          fittest_individual = individual

    result = []
    clique_size = 0
    for index in range(graph_len):
      # print(index)
      if fittest_individual[index] == 1:
        clique_size += 1
        result.append(index + 1)

    # print(result)
    # if True:
    #   for i in result:
    #     for j in result:
    #       if i != j:
    #         print(i,j)
    #         print(graph.has_edge(i, j))
    if generation % 10 == 0:
      print("Generation: ", generation, "Best_Fitness: ",
            best_fitness, "Individual: ", result, "Clique Size: ", clique_size)

  print('kurwa')
  # sum = 0 
  # fitness_array = []
  # for i in range(population_size):
  #     f = fitness(graph, population[i])
  #     fitness_array.append(f)
  #     sum += f
  # print(fitness_array)
  # print(sum)
  # while(best_fitness != 0 and population != 1000):



  print(graph.has_edge(2,1))
  # print(clique.max_clique(G))
  # print(len(clique.max_clique(G)))
  # print(clique.large_clique_size(G))
  # print(nx.find_cliques(G))
  # for c in nx.find_cliques(G):
  #   print(c) 
