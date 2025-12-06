import pandas as pd
import numpy as np
import math as math
import random
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import threading
import os


# read in coordinates from file to create dataset
def extract_coords(filename):
    data_names = ['xy', 'kilos','name']     # label columns
    data = pd.read_csv(filename, sep=', ', names=data_names, engine='python')

    data['xy'] = data['xy'].str.replace('[', '', regex=False)
    data['xy'] = data['xy'].str.replace(']', '', regex=False)
    xy_split = data['xy'].str.split(',', expand=True)
    data['x'] = pd.to_numeric(xy_split[0])
    data['y'] = pd.to_numeric(xy_split[1])
    data['kilos'] = data['kilos'].str.replace('{', '', regex=False)
    data['kilos'] = data['kilos'].str.replace('}', '', regex=False)
    data['kilos'] = pd.to_numeric(data['kilos'])

    if any(data['kilos'] > 9999):
        raise Exception("Error: Weight exceeds 9999.")
    
    
    return data

def create_matrix(data):
    # 8 x 12 matrix of [weights, names]
    rows=8
    cols=12
    matrix = [[(0,"") for c in range(cols)] for r in range(rows)]
    # print(matrix)

    for i in range(len(data)):
        x = 9 - int(data['x'].iloc[i])
        y = int(data['y'].iloc[i]) 
        if data['name'].iloc[i]=="NAN":
            weight=np.inf
        else:
            weight = int(data['kilos'].iloc[i])
        name = data['name'].iloc[i]
        matrix[x-1][y-1] = [weight, name]

    return matrix

class node:
    def __init__(self, grid):
        self.grid = grid
        self.depth = 0
        self.heuristicCost = 0
        self.moves = 0


def check_goal(grid):
    lweight = 0
    rweight = 0
    l_num_containers = 0
    r_num_containers = 0

    for row in grid:
        for i in range(0, len(row)//2):
            if row[i][1] != 'NAN' and row[i][1] != 'UNUSED':
                lweight += row[i][0]
                l_num_containers += 1
        
        for j in range(len(row)//2, len(row)):
            if row[j][1] != 'NAN' and row[j][1] != 'UNUSED':
                rweight += row[j][0]
                r_num_containers += 1

    # check for special case of 1 container or total 2 containers but 1 on each side
    if l_num_containers == 1 and r_num_containers == 0 or l_num_containers == 0 and r_num_containers == 1 or l_num_containers == 1 and r_num_containers == 1:
        return True

    # add check that diff is minimal!!

    return abs(lweight-rweight) < (lweight + rweight)*0.10 or abs(lweight-rweight) == 0


def general_search(grid, heuristic):
    priority_queue = []
    new_config = node(grid)
    new_config.heuristicCost = heuristic
    priority_queue.append(new_config)

    nodesVisited = 0
    maxQueueSize = 0
    visited = []    # list to store configurations that have been visited to prevent duplicate expansions

    visited.append(new_config.grid)

    while True:
        # sort queue by g(n) + h(n)
        priority_queue = sorted(priority_queue, key=lambda x: (x.depth + x.heuristicCost, x.depth))

        if not priority_queue:  # if queue is empty, failure
            print('Failure')
            return
        
        current = priority_queue[0]
        del priority_queue[0]

        # goal state is reached
        if check_goal(current.grid):
            print('Goal state!')
            print_grid(current.grid)
            return

        nodesVisited += 1

        print(f"best state to expand with g(n) = {current.depth} and h(n) = {current.heuristicCost} is {print_grid(current.grid)}")

        children = expand(current, visited)

        # update heuristics for children puzzle
        for child in children:
            child.heuristicCost = heuristic
            priority_queue.append(child)
            visited.append(child.grid)

        maxQueueSize = max(maxQueueSize, len(priority_queue))


def find_movable(grid):
    locs = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            # check spot is not empty
            if (grid[i][j][1] != 'UNUSED' and grid[i][j][1] != 'NAN'):
                # check there is no container on top
                if i == 0 or (i > 0 and grid[i-1][j][1] == 'UNUSED'): 
                    locs.append([i, j])

    return locs


def find_unused(grid):
    locs = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            # check space is empty
            if grid[i][j][1] == 'UNUSED':
                # check there is a container below or on bottom of grid (not "floating")
                if (i < len(grid)-1 and grid[i+1][j][1] != 'UNUSED') or (i == len(grid)-1):
                    locs.append([i, j])
    
    return locs


# generate children
def expand(parent, visited):
    children = []

    movable_locs = find_movable(parent.grid)
    unused_locs = find_unused(parent.grid)

    for [i, j] in movable_locs:
        for [k, l] in unused_locs:
            if [k, l] != [i-1, j]:
                new_grid = [row[:] for row in parent.grid]
                new_grid[i][j], new_grid[k][l] = new_grid[k][l], new_grid[i][j]
                child = node(new_grid)
                child.depth = parent.depth + 1
                children.append(child)

    for child in children:
        for grid in visited:
            if child.grid == grid:
                del child
                break
    
    return children

def a_star_heuristic(grid): 
    lweight = 0
    rweight = 0
    l_num_containers = []
    r_num_containers = []
    smallest_side = 0
    containers = []

    for row in grid:
        for i in range(0, len(row)//2):
            if row[i][1] != 'NAN' and row[i][1] != 'UNUSED':
                lweight += row[i][0]
                left_weight = row[i][0]
                l_num_containers.append(left_weight)
        
        for j in range(len(row)//2, len(row)):
            if row[j][1] != 'NAN' and row[j][1] != 'UNUSED':
                rweight += row[j][0]
                right_weight = row[j][0]
                r_num_containers.append(right_weight)
    # print("l_num_containers: ", l_num_containers)
    # print("r_num_containers: ", r_num_containers)
    balance_mass = (lweight + rweight) / 2
    # print("balance_mass: ", balance_mass)
    r_deficit = balance_mass - rweight
    # print("r_deficit", r_deficit)
    l_deficit = balance_mass - lweight
    # print("l_deficit", l_deficit)

    if(l_deficit > r_deficit):
        smallest_side = l_deficit
        containers = r_num_containers
    else:
        smallest_side = r_deficit
        containers = l_num_containers
    # print(f"smallest: {smallest_side}")
    # print(f"Containers: {containers}")
    descending_order = sorted(containers,reverse=True)
    # print(descending_order)
    hn = 0
    while smallest_side >= 0:
        for container in descending_order:
            if(container <= smallest_side):
                smallest_side-=container
                hn+=1
            else:
                continue
        break
    return hn


    

def print_grid(grid):
    for row in grid:
        for i in range(len(grid[0])):
            if row[i][1] == 'UNUSED':
                print(0, end=' ')
            elif row[i][1] == 'NAN':
                print('X', end=' ')
            else:
                print(row[i][0], end=' ')
        print( )


def main():

    print('\nbalance')

    input_file = input('Enter the name of file: ')
    if not( os.path.exists(input_file)):
        raise Exception("Warning: file does not exist, ending program.")
        
    file_name_w_ext = os.path.basename(input_file)
    file_name,ext = os.path.splitext(file_name_w_ext)
    
    data = extract_coords(input_file)
    grid = create_matrix(data)
    print_grid(grid)
    
    # general_search(grid,0)    
    general_search(grid,a_star_heuristic(grid))
    # a_star_heuristic(grid)


   
if __name__ == '__main__':
  main()