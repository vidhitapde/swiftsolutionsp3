
import pandas as pd
import _tkinter as tk
import numpy as np
import math as math
from pathlib import Path
import os
import heapq
import copy
import tkinter as tk
from datetime import date, datetime
import logging





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


def save_grid_as_input_format(grid, filename=""):
   rows = len(grid)
   cols = len(grid[0])
   lines = []


   for x in range(1, rows + 1):
       row_index = rows - x
       for y in range(1, cols + 1):
           col_index = y - 1
           weight, name = grid[row_index][col_index]


           if weight == np.inf or name in ("NAN", "UNUSED"):
               weight_str = "00000"
           else:
               weight_str = f"{int(weight):05}"


           lines.append(f"[{x:02},{y:02}], {{{weight_str}}}, {name}\n")


   with open(filename, "w") as f:
       f.writelines(lines)

   print(f"Grid saved to {filename}")
   


class node:
   def __init__(self, grid):
       self.grid = grid
       self.depth = 0
       self.heuristicCost = 0
       self.moves = 0
       self.parent = None
       self.craneLoc = [0,0]
       self.action = "start"
       self.sourceLoc = None
       self.targetLoc = None
       self.move = None
       self.cost = 0


   def __lt__(self, other):
       return (self.depth + self.heuristicCost) < (other.depth + other.heuristicCost)


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


   if l_num_containers == 1 and r_num_containers == 0 or l_num_containers == 0 and r_num_containers == 1 or l_num_containers == 1 and r_num_containers == 1:
       return True


   return abs(lweight-rweight) < (lweight + rweight)*0.10 or abs(lweight-rweight) == 0


def count_containers(grid):
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


   return (l_num_containers + r_num_containers)


def general_search(grid, heuristic):
   priority_queue = []
   visited = []


   new_config = node(grid)
   new_config.heuristicCost = heuristic
   heapq.heappush(priority_queue, new_config)
   visited.append(new_config)


   while True:
       if not priority_queue:
           print('Failure')
           return
      
       current = heapq.heappop(priority_queue)


       if check_goal(current.grid):
           print('Goal state!')
           print_grid(current.grid)
           print(f'Cost: {current.depth + (current.craneLoc[0] + current.craneLoc[1])}')
           return current


       children = expand(current, visited)


       for child in children:
           child.heuristicCost = a_star_heuristic(child.grid)
           heapq.heappush(priority_queue, child)
           visited.append(child.grid)




def find_movable(grid):
   locs = []
   for i in range(len(grid)):
       for j in range(len(grid[0])):
           if (grid[i][j][1] != 'UNUSED' and grid[i][j][1] != 'NAN'):
               if i == 0 or (i > 0 and grid[i-1][j][1] == 'UNUSED'):
                   locs.append([i, j])


   return locs




def find_unused(grid):
   locs = []
   for i in range(len(grid)):
       for j in range(len(grid[0])):
           if grid[i][j][1] == 'UNUSED':
               if (i < len(grid)-1 and grid[i+1][j][1] != 'UNUSED') or (i == len(grid)-1):
                   locs.append([i, j])
  
   return locs


def expand(parent, visited):
   children = []


   movable_locs = find_movable(parent.grid)
   unused_locs = find_unused(parent.grid)


   craney, cranex = parent.craneLoc[0], parent.craneLoc[1]


   # move crane only
   for [i, j] in movable_locs:
       if [craney, cranex] != [i,j]:
           child = node(parent.grid)
           child.parent = parent
           child.craneLoc = [i, j]
           child.depth = parent.depth + move_cost(parent.grid, [craney, cranex], [i,j])
           child.moves = parent.moves + 1
           child.cost = child.depth
           if [craney, cranex] == [0,0]:
               child.action = f"Move crane from PARK to {[abs(i-8), j+1]}"
           else:
               newy, newx = cranex+1, abs(craney-8)
               child.action = f"move crane from {[newx, newy]} to {[abs(i-8), j+1]}"
           children.append(child)


   # move container
   if [craney, cranex] in movable_locs:
       for [i, j] in unused_locs:
           if [i,j] != [craney-1, cranex]:
               new_grid = [row[:] for row in parent.grid]
               new_grid[craney][cranex], new_grid[i][j] = new_grid[i][j], new_grid[craney][cranex]
               child = node(new_grid)
               child.parent = parent
               child.craneLoc = [i,j]
               child.depth = parent.depth + move_cost(parent.grid, [craney, cranex], [i,j])
               child.moves = parent.moves + 1
               child.cost = child.depth
               moved_name = parent.grid[craney][cranex][1]
               child.action = f"move container '{moved_name}' from {[abs(craney-8), cranex+1]} to {[abs(i-8), j+1]}"
               child.sourceLoc = (craney, cranex)
               child.targetLoc = (i, j)
               child.move = ((craney, cranex), (i, j))
               children.append(child)


   filtered_children = []
   for child in children:
       is_visited = False
       for seen in visited:
           if child.grid == seen:
               is_visited = True
               break
       if not is_visited:
           filtered_children.append(child)
  
   return filtered_children


def print_steps(path):
   print(f"Solution has been found, it will take {path[-1].depth} minutes and {path[-1].moves} moves")
   for i, step in enumerate(path):
       if step.action != 'start':
           print(f"Step {i}: {step.action}")


def move_cost(grid, container, loc):
   tallest = find_height(grid, container, loc)


   if tallest == -1:
       return abs(container[0] - loc[0]) + abs(container[1] - loc[1])
   else:
       return abs(container[0] - (tallest-1)) + abs(container[1] - loc[1]) + abs((tallest-1) - loc[0])


def find_height(grid, container, loc):
   tallest = -1


   start_col, end_col = sorted((container[1], loc[1]))


   for row in range(0, container[0]+1):
       for col in range(start_col, end_col + 1):
           if grid[row][col][1] != 'UNUSED' and grid[row][col][1] != 'NAN' and [row, col] != container and [row, col] != loc:
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
  
   balance_mass = (lweight + rweight) / 2
   r_deficit = balance_mass - rweight
   l_deficit = balance_mass - lweight


   if(l_deficit > r_deficit):
       smallest_side = l_deficit
       containers = r_num_containers
   else:
       smallest_side = r_deficit
       containers = l_num_containers
   descending_order = sorted(containers,reverse=True)
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


def find_path(final_node):
   path = [final_node]
   curr = final_node
   while curr.parent:
       path.append(curr.parent)
       curr = curr.parent


   path.reverse()
  
   if final_node.craneLoc != [0,0]:
       end_config = node(final_node.grid)
       end_config.depth = final_node.depth + abs(final_node.craneLoc[0] + final_node.craneLoc[1])
       end_config.cost = end_config.depth
       end_config.craneLoc = [0, 0]
       end_config.parent = final_node
       end_config.moves = final_node.moves + 1
       end_config.action = f"move crane from {[abs(final_node.craneLoc[0]-8), final_node.craneLoc[1]+1]} to PARK"
       path.append(end_config)


   for grid in path:
       print(f"LENGTH: {len(path)}")
       print(f'Cost: {grid.depth}')
       print_grid_w_crane(grid)


   return path




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


def grid_to_coords(row, col):
   x = 9 - row
   y = col + 1
   return (x, y)


def print_grid_w_crane(node):
   grid = copy.deepcopy(node.grid)
   grid[node.craneLoc[0]][node.craneLoc[1]][0] = 'L'
   grid[node.craneLoc[0]][node.craneLoc[1]][1] = 'L'


   for row in grid:
       for i in range(len(grid[0])):
           if row[i][1] == 'UNUSED':
               print(0, end=' ')
           elif row[i][1] == 'NAN':
               print('X', end=' ')
           else:
               print(row[i][0], end=' ')
       print()


def create_log():
   today = date.today()
   now = datetime.now()
   formatted_date = today.strftime("%m %d %Y")
  
   time = datetime.now()
   formatted_time = time.strftime("_%H%M")
  
   os.makedirs("Logs", exist_ok=True)
  
   fileAddress = f'Logs/{formatted_date}{formatted_time}.txt'
  
   with open(fileAddress, "w") as f:
       f.write(f"{today.strftime('%m %d %Y')}: {time.strftime('%H:%M')} Program was started.\n")

   return fileAddress


def log_steps(path, filename, append=False):
   
   current_time = datetime.now()
  
   lines = []
  
   for i, step in enumerate(path):
       if step.action != 'start':
           timestamp = current_time.strftime("%Y-%m-%d %H:%M")
           lines.append(f"{timestamp} {i} of {len(path)-1}: {step.action}, {step.depth} minutes\n")
  
   lines.append("Done! Log was written.\n")
  
   mode = "a" if append else "w"
   with open(filename, mode) as f:
       f.writelines(lines)
  
   print(f"Log written to {filename}")


def log_steps_with_comments(path, filename, user_comments, step_timestamps, manifest_info=None, append=False):

   total_minutes = path[-1].depth
   total_moves = path[-1].moves
   current_time = datetime.now()
  
   lines = []
   
   lines.append(f"{current_time.strftime('%m %d %Y')}: {current_time.strftime('%H:%M')} Program was started.\n")
   
   if manifest_info:
       lines.append(f"{current_time.strftime('%m %d %Y')}: {current_time.strftime('%H:%M')} Manifest {manifest_info['filename']} is opened, there are {manifest_info['container_count']} containers on the ship.\n")
       lines.append(f"{current_time.strftime('%m %d %Y')}: {current_time.strftime('%H:%M')} Balance solution found, it will require {manifest_info['total_moves']} moves/{manifest_info['total_minutes']} minutes.\n")
   
   lines.append(f"\nSolution was found, it will take {total_minutes} minutes and {total_moves} moves\n\n")
  
   for i, step in enumerate(path):
       if step.action != 'start':
           if i in step_timestamps:
               timestamp = step_timestamps[i].strftime("%m %d %Y: %H:%M")
           else:
               timestamp = current_time.strftime("%m %d %Y: %H:%M")
               
           lines.append(f"{timestamp} Step {i} of {len(path)-1}: {step.action}, {step.depth} minutes\n")
           
           if i in user_comments:
               for comment_data in user_comments[i]:
                   lines.append(f"{comment_data['timestamp']} {comment_data['comment']}\n")
           lines.append("\n")
       else:
           if 0 in step_timestamps:
               timestamp = step_timestamps[0].strftime('%m %d %Y: %H:%M')
           else:
               timestamp = current_time.strftime("%m %d %Y: %H:%M")
           lines.append(f"{timestamp} Program started\n")
           
           if 0 in user_comments:
               for comment_data in user_comments[0]:
                   lines.append(f"{comment_data['timestamp']} {comment_data['comment']}\n")
           lines.append("\n")
  
   lines.append("Done! Log was written.\n")
  
   mode = "a" if append else "w"
   with open(filename, mode) as f:
       f.writelines(lines)
  
   print(f"Enhanced log with real-time timestamps written to {filename}")


CELL_SIZE = 90
ROWS = 8
COLS = 12


COLORS = {
   "UNUSED": "white",
   "NAN": "gray",
}


class GridGUI:
   def __init__(self, path, logAddress, manifest_info=None):
       self.path = path
       self.index = 0
       self.logAddress = logAddress
       self.user_comments = {} 
       self.step_timestamps = {} 
       self.manifest_info = manifest_info 
       self.logged_steps = set() 
      
       self.root = tk.Tk()
       self.root.title("Ship Container Balancer Visualization")
      
       self.canvas = tk.Canvas(self.root, width=COLS*CELL_SIZE, height=ROWS*CELL_SIZE)
       self.canvas.pack()
      
       # Add text display for moves
       self.move_label = tk.Label(self.root, text="", font=("Arial", 12), justify="left", wraplength=700)
       self.move_label.pack(pady=10)
      
       self.root.bind("<Return>", self.next_step)
      
       button_frame = tk.Frame(self.root)
       button_frame.pack(pady=5)
      
       self.log_button = tk.Button(button_frame, text="Save Log to File", command=self.write_log)
       self.log_button.pack(side="left", padx=5)
      
       self.comment_button = tk.Button(button_frame, text="Add Comment to Log", command=self.add_comment)
       self.comment_button.pack(side="left", padx=5)
      
       self.label = tk.Label(self.root, text="Press Enter to advance to the next step")
       self.label.pack(pady=5)
      
       self.step_timestamps[0] = datetime.now()
       self.logged_steps.add(0)
       
       self.update_display()
       self.root.mainloop()
  
   def write_log(self):
    import shutil
    from tkinter import filedialog, messagebox
    from datetime import datetime

    timestamp = datetime.now().strftime("%b_%d_%Y_%H%M")
    default_name = f"KeoghsPort_{timestamp}.txt"

    filename = filedialog.asksaveasfilename(
        initialfile=default_name,
        defaultextension=".txt",
        filetypes=[("Text files", "*.txt")]
    )

    if filename:
        try:
            shutil.copyfile(self.logAddress, filename)
            messagebox.showinfo("Success", "Log file saved successfully!")
            print(f"Log copied to: {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save log: {e}")

  
   def add_comment(self):
       from tkinter import simpledialog, messagebox
      
       comment = simpledialog.askstring(
           "Add Comment",
           "Enter your comment:",
           parent=self.root
       )
      
       if comment:
           try:
               time = datetime.now()
               formatted_time = time.strftime("%H:%M")
               
               # Store comment for inclusion in downloaded log
               if self.index not in self.user_comments:
                   self.user_comments[self.index] = []
               self.user_comments[self.index].append({
                   'comment': comment,
                   'timestamp': formatted_time
               })
              
               try:
                    with open(self.logAddress, "r", encoding="utf-8") as f:
                        lines = f.readlines()
               except FileNotFoundError:
                    with open(self.logAddress, "w", encoding="utf-8") as f:
                        if self.index == 0:
                            f.write(f"{time.strftime('%Y-%m-%d %H:%M')} Program was started\n")
                            f.write(f"{formatted_time} {comment}\n\n")
                        else:
                            current_node = self.path[self.index]
                            step_timestamp = self.step_timestamps.get(self.index, time)
                            f.write(f"{step_timestamp.strftime('%Y-%m-%d %H:%M')} Step {self.index} of {len(self.path)-1}: {current_node.action}, {current_node.depth} minutes\n")
                            f.write(f"{formatted_time} {comment}\n\n")
               else:
                    insert_idx = None
                    search_token = None
                    if self.index == 0:
                        search_token = "Program was started"
                    else:
                        search_token = f" {self.index} of {len(self.path)-1}:"

                    for i, line in enumerate(lines):
                        if search_token in line:
                            insert_idx = i + 1
                            break

                    comment_lines = []
                    if self.index == 0:
                        comment_lines.append(f"{time.strftime('%Y-%m-%d %H:%M')} Program was started\n")
                        comment_lines.append(f"{formatted_time} {comment}\n\n")
                    else:
                        current_node = self.path[self.index]
                        step_timestamp = self.step_timestamps.get(self.index, time)
                        comment_lines.append(f"{step_timestamp.strftime('%Y-%m-%d %H:%M')} Step {self.index} of {len(self.path)-1}: {current_node.action}, {current_node.depth} minutes\n")
                        comment_lines.append(f"{formatted_time} {comment}\n\n")

                    if insert_idx is None:
                        lines.extend(comment_lines)
                    else:
                        for offset, cl in enumerate(comment_lines):
                            lines.insert(insert_idx + offset, cl)

                    with open(self.logAddress, "w", encoding="utf-8") as f:
                        f.writelines(lines)
              
               print(f"Comment added to log: {comment}")
               messagebox.showinfo("Success", "Comment added to log file!")
           except Exception as e:
               messagebox.showerror("Error", f"Could not write to log: {e}")
  
   def draw_grid(self, node, source_loc=None, target_loc=None):
       self.canvas.delete("all")
       grid = node.grid
       for r in range(ROWS):
           for c in range(COLS):
               weight, name = grid[r][c]
              
               if source_loc and source_loc == (r, c):
                   color = "green"  
               elif target_loc and target_loc == (r, c):
                   color = "red"
               elif name == "UNUSED":
                   color = "white"
               elif name == "NAN":
                   color = "gray"
               else:
                   color = "tan"
                  
               x1, y1 = c*CELL_SIZE, r*CELL_SIZE
               x2, y2 = x1+CELL_SIZE, y1+CELL_SIZE
               self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")
               if name not in ("UNUSED", "NAN"):
                   self.canvas.create_text(x1+CELL_SIZE/2, y1+CELL_SIZE/2, text=name, font=("Arial", 10, "bold"))
      
       cr = node.craneLoc
       self.canvas.create_rectangle(cr[1]*CELL_SIZE, cr[0]*CELL_SIZE, (cr[1]+1)*CELL_SIZE, (cr[0]+1)*CELL_SIZE,
                                    outline="blue", width=4)
  
   def update_display(self):
    current_node = self.path[self.index]

    final_cost = self.path[-1].depth
    final_moves = self.path[-1].moves

    source_loc = current_node.sourceLoc
    target_loc = current_node.targetLoc

    self.draw_grid(current_node, source_loc, target_loc)

    move_text = f"Step {self.index} of {len(self.path)-1}\n"
    move_text += f"Total solution cost: {final_cost} minutes, {final_moves} moves\n\n"

    if current_node.action != "start":
        move_text += f"{current_node.action}\n"
        move_text += f"Time spent on this step: {current_node.depth - current_node.parent.depth} minutes\n"
    else:
        move_text += "Initial state\n"

    self.move_label.config(text=move_text)

    print(f"\n=== Step {self.index} ===")
    print(move_text)

    try:
        if self.index not in self.logged_steps:
            self._write_step_to_log(self.index)
            self.logged_steps.add(self.index)
    except Exception as e:
        print(f"Could not write step {self.index} to log: {e}")

  
   def next_step(self, event):
       if self.index < len(self.path)-1:
           self.index += 1
           self.step_timestamps[self.index] = datetime.now()
           self.update_display()
       else:
           try:
               from tkinter import messagebox
               outfile = self.manifest_info.get('output_filename') if self.manifest_info else None
               if outfile is None:
                   if self.manifest_info and 'filename' in self.manifest_info:
                       base = os.path.splitext(self.manifest_info.get('filename'))[0]
                       outfile = f"{base}_OUTBOUND.txt"
                   else:
                       outfile = 'OUTBOUND.txt'

               finished_msg = (
                   f"Finished a cycle. Manifest \"{outfile}\" was written to desktop, "
                   "and a reminder pop-up to operator to send file was displayed."
               )

               timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
               try:
                   with open(self.logAddress, 'a', encoding='utf-8') as f:
                       f.write(f"{timestamp} {finished_msg}\n")
               except Exception as e:
                   print(f"Could not write finished-cycle message to log: {e}")

               messagebox.showinfo("Send Manifest", f"Please send manifest file: {outfile} to the operator.")
           except Exception as e:
               print(f"Error handling end-of-path actions: {e}")
           print("End of path reached.")

   def _write_step_to_log(self, index):
    
        current_time = self.step_timestamps.get(index, datetime.now())
        timestamp = current_time.strftime('%Y-%m-%d %H:%M')
        try:
            with open(self.logAddress, 'a', encoding='utf-8') as f:
                if index == 0:
                    f.write(f"{timestamp} Program was started\n")
                else:
                    node = self.path[index]
                    f.write(f"{timestamp} Step {index} of {len(self.path)-1}: {node.action}, {node.depth} minutes\n")
        except Exception:
            raise


def main():
   print('\nbalance')
  
   logAddress = create_log()
  
   input_file = input('Enter the name of file: ')
   if not os.path.exists(input_file):
       raise Exception("Warning: file does not exist, ending program.")
  
   file_name_w_ext = os.path.basename(input_file)
   file_name, ext = os.path.splitext(file_name_w_ext)
  
   data = extract_coords(input_file)
   grid = create_matrix(data)
  
   with open(logAddress, "a") as f:
       current_time = datetime.now()
       f.write(f"{current_time.strftime('%m %d %Y')}: {current_time.strftime('%H:%M')} Manifest {file_name_w_ext} is opened, there are {count_containers(grid)} containers on the ship.\n")
   
   final_node = general_search(grid, a_star_heuristic(grid))
   solution_path = find_path(final_node)

   with open(logAddress, "a") as f:
       solution_time = datetime.now()
       total_moves = solution_path[-1].moves
       total_minutes = solution_path[-1].depth
       f.write(f"{solution_time.strftime('%m %d %Y')}: {solution_time.strftime('%H:%M')} Balance solution found, it will require {total_moves} moves/{total_minutes} minutes.\n")

   print_steps(solution_path)
   
   balanced_grid = final_node.grid
   output_filename = f"{file_name}_OUTBOUND.txt"
   save_grid_as_input_format(balanced_grid, output_filename)
   
   outbound_filename = "OUTBOUND.txt"
   save_grid_as_input_format(balanced_grid, outbound_filename)
   
   with open(logAddress, "a") as f:
       file_creation_time = datetime.now()
       f.write(f"{file_creation_time.strftime('%m %d %Y')}: {file_creation_time.strftime('%H:%M')} Balanced manifest '{output_filename}' was written to disk.\n")
       f.write(f"{file_creation_time.strftime('%m %d %Y')}: {file_creation_time.strftime('%H:%M')} Balanced manifest '{outbound_filename}' was written to disk.\n")
  
  
   manifest_info = {
       'filename': file_name_w_ext,
       'container_count': count_containers(grid),
       'total_moves': solution_path[-1].moves,
       'total_minutes': solution_path[-1].depth,
       'output_filename': output_filename
   }
   GridGUI(solution_path, logAddress, manifest_info)


if __name__ == '__main__':
   main()
