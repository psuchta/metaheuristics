import numpy as np
import math
import random
import geopy.distance

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

def find_closest(chosen_city_index, cities, visited_cities):
  global global_distance
  chosen_city = cities[chosen_city_index - 1]
  chosen_geo = (float(chosen_city[1]), float(chosen_city[2]))

  min_distance = math.inf
  current_city_index = None
  
  # Loop throught all cities and pick nearest city to the chosen one
  for city in cities: 
    if int(city[0]) in visited_cities:
      continue

    city_geo = (float(city[1]), float(city[2]))
    distance = math.dist(chosen_geo, city_geo)
    if distance < min_distance:
      min_distance = distance
      min_city_index = int(city[0])
  global_distance += min_distance
  return min_city_index

if __name__ == '__main__':
  cities, city_count = load_cities('ch130.txt')
  city_count = int(city_count)

  picked_city = random.randint(1, city_count)
  visited_cities = []

  for index in range(len(cities) - 1):
    visited_cities.append(picked_city)
    picked_city = find_closest(picked_city, cities, visited_cities)
    # print(picked_city)
  visited_cities.append(picked_city)
  print(f'Chosen path: {visited_cities}')
  print(f'Global distance: {global_distance}')
