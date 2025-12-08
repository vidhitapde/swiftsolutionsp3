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
import heapq
import tkinter as tk


# ------------ DATA / SEARCH CODE ------------

# read in coordinates from file to create dataset
def extract_coords(filename):
    data_names = ['xy', 'kilos', 'name']     # label columns
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
    rows = 8
    cols = 12
    matrix = [[(0, "") for c in range(cols)] for r in range(rows)]
    # print(matrix)

    for i in range(len(data)):
        x = 9 - int(data['x'].iloc[i])
        y = int(data['y'].iloc[i])
        if data['name'].iloc[i] == "NAN":
            weight = np.inf
        else:
            weight = int(data['kilos'].iloc[i])
        name = data['name'].iloc[i]
        matrix[x - 1][y - 1] = [weight, name]

    return matrix


class node:
    def __init__(self, grid):
        self.grid = grid
        self.depth = 0
        self.heuristicCost = 0
        self.moves = 0
        self.parent = None
        self.craneLoc = [0, 0]
        self.move = None

    def __lt__(self, other):
        return (self.depth + self.heuristicCost) < (other.depth + other.heuristicCost)

    def __lt__(self, other):
        return (self.depth + self.heuristicCost) < (other.depth + other.heuristicCost)

def check_goal(grid):
    lweight = 0
    rweight = 0
    l_num_containers = 0
    r_num_containers = 0

    for row in grid:
        for i in range(0, len(row) // 2):
            if row[i][1] != 'NAN' and row[i][1] != 'UNUSED':
                lweight += row[i][0]
                l_num_containers += 1

        for j in range(len(row) // 2, len(row)):
            if row[j][1] != 'NAN' and row[j][1] != 'UNUSED':
                rweight += row[j][0]
                r_num_containers += 1

    # check for special case of 1 container or total 2 containers but 1 on each side
    if (l_num_containers == 1 and r_num_containers == 0) or \
       (l_num_containers == 0 and r_num_containers == 1) or \
       (l_num_containers == 1 and r_num_containers == 1):
        return True

    # add check that diff is minimal!!
    return abs(lweight - rweight) < (lweight + rweight) * 0.10 or abs(lweight - rweight) == 0


def general_search(grid, heuristic):
    priority_queue = []
    visited = []    # list to store configurations that have been visited to prevent duplicate expansions

    new_config = node(grid)
    new_config.heuristicCost = heuristic
    heapq.heappush(priority_queue, new_config)
    visited.append(new_config)

    while True:
        if not priority_queue:  # if queue is empty, failure
            print('Failure')
            return None

        current = heapq.heappop(priority_queue)

        # goal state is reached
        if check_goal(current.grid):
            print('Goal state!')
            print_grid(current.grid)
            cost = current.depth + (current.craneLoc[0] + current.craneLoc[1])
            print(f'Cost: {cost}')
            current.cost = cost
            return current  

        print(f"best state to expand with g(n) = {current.depth} and h(n) = {current.heuristicCost} is ")
        print_grid(current.grid)

        children = expand(current, visited)

        # update heuristic for children puzzle
        for child in children:
            child.heuristicCost = a_star_heuristic(child.grid)
            heapq.heappush(priority_queue, child)
            visited.append(child.grid)


def find_movable(grid):
    locs = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            # check spot is not empty
            if (grid[i][j][1] != 'UNUSED' and grid[i][j][1] != 'NAN'):
                # check there is no container on top
                if i == 0 or (i > 0 and grid[i - 1][j][1] == 'UNUSED'):
                    locs.append([i, j])

    return locs


def find_unused(grid):
    locs = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            # check space is empty
            if grid[i][j][1] == 'UNUSED':
                # check there is a container below or on bottom of grid (not "floating")
                if (i < len(grid) - 1 and grid[i + 1][j][1] != 'UNUSED') or (i == len(grid) - 1):
                    locs.append([i, j])

    return locs


# generate children
def expand(parent, visited):
    children = []

    movable_locs = find_movable(parent.grid)
    unused_locs = find_unused(parent.grid)

    craney, cranex = parent.craneLoc[0], parent.craneLoc[1]

    # move crane only
    for [i, j] in movable_locs:
        if [craney, cranex] != [i, j]:    # crane is not at movable container
            child = node(parent.grid)
            child.parent = parent
            child.craneLoc = [i, j]
            child.depth = parent.depth + abs(parent.craneLoc[0] - i) + abs(parent.craneLoc[1] - j)
            child.moves = parent.moves + 1
            child.move = None  
            children.append(child)

    if [craney, cranex] in movable_locs:
        for [i, j] in unused_locs:
            if [i, j] != [craney - 1, cranex]:  # do not drop on top of current container
                new_grid = [row[:] for row in parent.grid]
                new_grid[craney][cranex], new_grid[i][j] = new_grid[i][j], new_grid[craney][cranex]
                child = node(new_grid)
                child.parent = parent
                child.craneLoc = [i, j]
                child.depth = parent.depth + move_cost(parent.grid, [craney, cranex], [i, j])
                child.moves = parent.moves + 1
                child.move = ((craney, cranex), (i, j))
                children.append(child)

    for child in children:
        for seen in visited:
            if child.grid == seen:
                del child
                break

    return children


def move_cost(grid, container, loc):
    tallest = find_height(grid, container, loc)

    if tallest == -1:   # case 1: no obstacle
        return abs(container[0] - loc[0]) + abs(container[1] - loc[1])     # manhattan dist b/w container and loc
    else:
        # move up above tallest container + across to loc + down to loc
        # (tallest - 1) to clear container
        return abs(container[0] - (tallest - 1)) + abs(container[1] - loc[1]) + abs((tallest - 1) - loc[0])


# find tallest height of any container between container to be moved and the final location
def find_height(grid, container, loc):
    tallest = -1

    start_col, end_col = sorted((container[1], loc[1]))

    for row in range(0, container[0] + 1):      # y-axis from top to height of container decrementing by 1
        for col in range(start_col + 1, end_col + 1):    # x-axis between container and location
            if grid[row][col][1] != 'UNUSED' and grid[row][col][1] != 'NAN':
                # print(grid[row][col][1])
                # print('lllllll'
                return row

    return tallest


def a_star_heuristic(grid): 
    lweight = 0
    rweight = 0
    l_num_containers = []
    r_num_containers = []
    smallest_side = 0
    containers = []

    for row in grid:
        for i in range(0, len(row) // 2):
            if row[i][1] != 'NAN' and row[i][1] != 'UNUSED':
                lweight += row[i][0]
                left_weight = row[i][0]
                l_num_containers.append(left_weight)
        
        for j in range(len(row) // 2, len(row)):
            if row[j][1] != 'NAN' and row[j][1] != 'UNUSED':
                rweight += row[j][0]
                right_weight = row[j][0]
                r_num_containers.append(right_weight)

    balance_mass = (lweight + rweight) / 2
    r_deficit = balance_mass - rweight
    l_deficit = balance_mass - lweight

    if (l_deficit > r_deficit):
        smallest_side = l_deficit
        containers = r_num_containers
    else:
        smallest_side = r_deficit
        containers = l_num_containers

    descending_order = sorted(containers, reverse=True)
    hn = 0
    while smallest_side >= 0:
        for container in descending_order:
            if (container <= smallest_side):
                smallest_side -= container
                hn += 1
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
        print()
    print()


def reconstruct_path(goal_node):
    path = []
    node_ptr = goal_node
    while node_ptr is not None:
        path.append(node_ptr)
        node_ptr = node_ptr.parent
    path.reverse()
    return path


# ------------ TKINTER VISUALIZER ------------

class Visualizer:
    def __init__(self, root, path_nodes):
        self.root = root
        self.path_nodes = path_nodes

        self.moves = [n.move for n in path_nodes[1:]]  
        self.num_moves = len(self.moves)               
        self.total_actions = self.num_moves + 2        
        self.current_step = 0                          

        self.rows = 8
        self.cols = 12
        self.cell_size = 40
        self.grid_top_y = 120
        self.crane_y = 40

        self.history_lines = []

        
        self.text_label = tk.Label(root, font=("Arial", 16), justify="left")
        self.text_label.pack(pady=10)

        canvas_width = self.cols * self.cell_size + 40
        canvas_height = self.grid_top_y + self.rows * self.cell_size + 40
        self.canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
        self.canvas.pack()

        self.cell_rects = [[None for _ in range(self.cols)] for _ in range(self.rows)]
        self.cell_texts = [[None for _ in range(self.cols)] for _ in range(self.rows)]
        for r in range(self.rows):
            for c in range(self.cols):
                x1 = 20 + c * self.cell_size
                y1 = self.grid_top_y + r * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                rect = self.canvas.create_rectangle(x1, y1, x2, y2,
                                                    outline="black", fill="white")
                self.cell_rects[r][c] = rect

                text = self.canvas.create_text(
                    (x1 + x2) // 2, (y1 + y2) // 2,
                    text="", font=("Arial", 10), fill="black"
                )
                self.cell_texts[r][c] = text

        # crane "park" position (left side above grid)
        self.park_x1 = 20
        crane_x1 = self.park_x1
        crane_y1 = self.crane_y
        crane_x2 = crane_x1 + self.cell_size
        crane_y2 = crane_y1 + self.cell_size
        self.crane_rect = self.canvas.create_rectangle(
            crane_x1, crane_y1, crane_x2, crane_y2,
            outline="black", fill="lightgrey"
        )

        self.draw_grid_for_node(0, src=None, dst=None)

        # initial message
        self.text_label.config(
            text=(
                f"Solution has been found, it will take {self.total_actions} move(s)\n"
                f"Hit ENTER when ready for first move"
            )
        )

        self.root.bind("<Return>", self.next_step)

    def grid_colors_from_node(self, node_index, src=None, dst=None):
        """Return 2D list of colors for the grid cells at given node."""
        grid = self.path_nodes[node_index].grid
        colors = []
        for i in range(self.rows):
            row_colors = []
            for j in range(self.cols):
                name = grid[i][j][1]
                if name == 'NAN':
                    color = "black"
                elif name == 'UNUSED' or name == "":
                    color = "white"
                else:
                    color = "#a1937a"  # light brown / tan for containers feel (free to change)
                row_colors.append(color)
            colors.append(row_colors)

        # override source/target cells if they exist (for grid cells only) (also feel free to change colors if needed)
        if src is not None:
            si, sj = src
            colors[si][sj] = "green"
        if dst is not None:
            di, dj = dst
            colors[di][dj] = "red"

        return colors

    def draw_grid_for_node(self, node_index, src=None, dst=None):
        colors = self.grid_colors_from_node(node_index, src=src, dst=dst)
        grid = self.path_nodes[node_index].grid

        for i in range(self.rows):
            for j in range(self.cols):
                self.canvas.itemconfig(self.cell_rects[i][j], fill=colors[i][j])

                name = grid[i][j][1]
                label = ""
                if name not in ("UNUSED", "NAN", "", None):
                    label = name.split()[0]  # first word only
                self.canvas.itemconfig(self.cell_texts[i][j], text=label)

        if dst is not None:
            col = dst[1]
            x1 = 20 + col * self.cell_size
            y1 = self.crane_y
            x2 = x1 + self.cell_size
            y2 = y1 + self.cell_size
            self.canvas.coords(self.crane_rect, x1, y1, x2, y2)

    def next_step(self, event=None):
        N = self.num_moves
        A = self.total_actions  

        if self.current_step > N + 2:
            return

        if self.current_step == N + 2:
            goal_cost = getattr(self.path_nodes[-1], "cost", None)
            self.draw_grid_for_node(N, src=None, dst=None)
            self.canvas.coords(
                self.crane_rect,
                self.park_x1, self.crane_y,
                self.park_x1 + self.cell_size, self.crane_y + self.cell_size
            )
            self.canvas.itemconfig(self.crane_rect, fill="lightgrey")
            self.text_label.config(
                text=f"Final container placements, with a cost of {goal_cost}\nDone!"
            )
            self.current_step += 1
            return

        if self.current_step == N + 1:
            (_, _), (last_di, last_dj) = self.moves[-1]

            self.draw_grid_for_node(N, src=(last_di, last_dj), dst=None)

            self.canvas.coords(
                self.crane_rect,
                self.park_x1, self.crane_y,
                self.park_x1 + self.cell_size, self.crane_y + self.cell_size
            )
            self.canvas.itemconfig(self.crane_rect, fill="red")

            step_num = self.current_step + 1  
            dst_row = self.rows - last_di
            dst_col = last_dj + 1
            dst_str = f"[{dst_row:02},{dst_col:02}]"

            line = f"{step_num} of {A}: Move crane from {dst_str} to park"
            self.history_lines.append(line)
            self.text_label.config(
                text="\n".join(self.history_lines) + "\nHit ENTER when done"
            )

            self.current_step += 1
            return

        if self.current_step == 0:
            if N == 0:
                self.text_label.config(text="No moves needed.\nDone!")
                self.current_step = N + 3
                return

            (si, sj), _ = self.moves[0]

            self.draw_grid_for_node(0, src=None, dst=(si, sj))

            x1 = 20 + sj * self.cell_size
            y1 = self.crane_y
            x2 = x1 + self.cell_size
            y2 = y1 + self.cell_size
            self.canvas.coords(self.crane_rect, x1, y1, x2, y2)
            self.canvas.itemconfig(self.crane_rect, fill="green")

            step_num = 1
            dst_row = self.rows - si
            dst_col = sj + 1
            dst_str = f"[{dst_row:02},{dst_col:02}]"

            line = f"{step_num} of {A}: Move the crane from park to {dst_str}"
            self.history_lines.append(line)
            self.text_label.config(
                text="\n".join(self.history_lines) + "\nHit ENTER when done"
            )

            self.current_step += 1
            return

        k = self.current_step          
        move = self.moves[k - 1]
        (si, sj), (di, dj) = move

        node_index = k
        self.draw_grid_for_node(node_index, src=(si, sj), dst=(di, dj))

        self.canvas.itemconfig(self.crane_rect, fill="lightgrey")

        step_num = self.current_step + 1 
        src_row = self.rows - si
        src_col = sj + 1
        dst_row = self.rows - di
        dst_col = dj + 1
        src_str = f"[{src_row:02},{src_col:02}]"
        dst_str = f"[{dst_row:02},{dst_col:02}]"

        line = f"{step_num} of {A}: Move from {src_str} to {dst_str}"
        self.history_lines.append(line)
        self.text_label.config(
            text="\n".join(self.history_lines) + "\nHit ENTER when done"
        )

        self.current_step += 1


def run_visualization(path_nodes):
    root = tk.Tk()
    root.title("Container Balancing Visualization")
    Visualizer(root, path_nodes)
    root.mainloop()


# ------------ MAIN ------------

def main():
    print('\nbalance')

    input_file = input('Enter the name of file: ')
    if not (os.path.exists(input_file)):
        raise Exception("Warning: file does not exist, ending program.")

    data = extract_coords(input_file)
    grid = create_matrix(data)
    print("Initial grid:")
    print_grid(grid)

    goal = general_search(grid, a_star_heuristic(grid))
    if goal is None:
        print("No solution found.")
        return

    full_path = reconstruct_path(goal)  

    container_nodes = [n for n in full_path[1:] if n.move is not None]
    path_nodes = [full_path[0]] + container_nodes

    print(f"Solution has {len(path_nodes) - 1} container move(s).")

    # run Tkinter visualization
    run_visualization(path_nodes)


if __name__ == '__main__':
    main()