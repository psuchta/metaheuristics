import numpy as np
import math
import random
import geopy.distance
from numpy.random import choice

global_distance = 0

def load_cities(file_name):
  city_count = 0
  cities = []
  with open(file_name) as f:
    for index, line in enumerate(f):
      splited_line = line.strip().split()
      if index == 0:
        city_count = splited_line[0]
        continue
      cities.append(splited_line)
  return cities, city_count

# Remove visited cities from all cities array
def cities_without_visited(visited_cities, cities):
  new_cities = []

  for city in cities: 
    if int(city[0]) in visited_cities:
      continue
    new_cities.append(city)
  return new_cities

# Calculate sum of distances between chosen city and other non visited ones
def sum_of_distances(chosen_city, cities):
  chosen_geo = (float(chosen_city[1]), float(chosen_city[2]))
  distance_sum = 0

  for city in cities: 
    city_geo = (float(city[1]), float(city[2]))
    distance = math.dist(chosen_geo, city_geo)
    distance_sum += 1/distance**2
  return distance_sum

def calculate_distance_probability(distance_sum, chosen_city, second_city):
  # print(chosen_city)
  chosen_geo = (float(chosen_city[1]), float(chosen_city[2]))
  second_city_geo = (float(second_city[1]), float(second_city[2]))
  distance = math.dist(chosen_geo, second_city_geo)

  probability = (1/distance**2)/distance_sum
  return probability


def find_closest(chosen_city_index, cities, visited_cities):
  global global_distance

  chosen_city = cities[chosen_city_index - 1]
  chosen_geo = (float(chosen_city[1]), float(chosen_city[2]))
  probabilities = []

  not_visited_cities = cities_without_visited(visited_cities, cities) 

  min_distance = math.inf
  current_city_index = None
    
  distance_sum = sum_of_distances(chosen_city, not_visited_cities) 

  # Loop throught all cities and calculate their probability.
  # If current city is closer to the chosen city, probability of selecting is greater.
  for city in not_visited_cities: 
    probabilities.append(calculate_distance_probability(distance_sum, chosen_city, city))

  cities_indexes = [row[0] for row in not_visited_cities]
  # min_city_index = int(random.choices(population=cities_indexes, weights = probabilities, k=1)[0])
  min_city_index = int(np.random.choice(cities_indexes, p=probabilities))
  
  closest_city = cities[min_city_index - 1]
  city_geo = (float(closest_city[1]), float(closest_city[2]))
  distance = math.dist(chosen_geo, city_geo)
  # Add distance between points to global_distance
  global_distance += distance
  return min_city_index

if __name__ == '__main__':
  cities, city_count = load_cities('ch130.txt')
  city_count = int(city_count)

  picked_city = random.randint(1, city_count)
  visited_cities = []

  # Loop as many times as there are points in the file - 1 
  for index in range(len(cities) - 1):
    visited_cities.append(picked_city)
    picked_city = find_closest(picked_city, cities, visited_cities)

  visited_cities.append(picked_city)
  print(f'Chosen path: {visited_cities}')
  print(f'Global distance: {global_distance}')
