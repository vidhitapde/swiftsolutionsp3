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
        x = int(data['x'].iloc[i])
        y = int(data['y'].iloc[i])
        if data['name'].iloc[i]=="NAN":
            weight=np.inf
        else:
            weight = data['kilos'].iloc[i]
        name = data['name'].iloc[i]
        matrix[x-1][y-1] = [weight, name]

    return matrix

def main():

    print('\nbalance')

    input_file = input('Enter the name of file: ')
    if not( os.path.exists(input_file)):
        raise Exception("Warning: file does not exist, ending program.")
        
    file_name_w_ext = os.path.basename(input_file)
    file_name,ext = os.path.splitext(file_name_w_ext)
    
    data = extract_coords(input_file)
    dist_matrix = create_matrix(data)
    print(dist_matrix)

   
if __name__ == '__main__':
  main()