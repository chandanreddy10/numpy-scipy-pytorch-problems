import numpy as np
import matplotlib.pyplot as plt
import os 
from pathlib import Path 

START=0
END_AT=100
DATA_DIR="misc"
MEAN=0
VARIANCE=1
X= 5
Y=5 

def elementary_random_walk(start=START,end=END_AT):
    path = []
    old_step=START
    index= 0 
    while old_step < end:
        print(f"Step Count : {index} | Value : {old_step}")
        path.append(old_step)
        next_step = np.random.choice([1, -1], p=[0.5, 0.5])
        old_step = old_step + next_step
    
    return path

def gauss_random_walk(start=START,end=END_AT, mean=MEAN, variance=VARIANCE):
    path = []
    old_step=START
    index= 0 
    while old_step < end:
        print(f"Step Count : {index} | Value : {old_step}")
        path.append(old_step)
        next_step = np.random.normal(MEAN, VARIANCE)
        old_step = old_step + next_step
    
    return path

def plot_random_walk(path, filename="random_walk"):
    x=np.arange(0, len(path))
    
    plt.plot(x, path)
    plt.xlabel("Steps")
    plt.ylabel("Value")
    plt.title(f"Random walk steps : {len(path)}")
    plt.savefig(f"misc/{filename}.jpg")
    plt.show()
    print("Saved Fig")

def generate_grid(x=X, y=Y):

    x = np.arange(0,x)
    y = np.arange(0,y)

    X, Y = np.meshgrid(x, y)
    return X, Y

def plot_grid(X, Y, start_loc=(0,0), end_loc=(4,4)):
    start_x, start_y = start_loc[0], start_loc[1]
    end_x, end_y = end_loc[0], end_loc[1]

    plt.figure(figsize=(6,6))
    plt.scatter(X, Y, color='lightgray')
    plt.scatter(start_x, start_y, c='green', s=200, label='Start')
    plt.scatter(end_x,end_y, c='red', s=200, label='Goal')
    plt.grid(True)
    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.title("Grid with np.meshgrid")
    plt.show()
### Moving the Project to bigger one.
#Random Walks as City Navigation
# path = elementary_random_walk()
# plot_random_walk(path)
# path = gauss_random_walk()
# plot_random_walk(path, filename="gauss_random_walk")
X, Y = generate_grid()
plot_grid(X, Y)