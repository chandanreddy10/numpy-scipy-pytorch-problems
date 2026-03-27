import numpy as np
import matplotlib.pyplot as plt
import os 
from pathlib import Path 

START=0
END_AT=10
DATA_DIR="misc"

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

def plot_random_walk(path):
    x=np.arange(0, len(path))
    
    plt.plot(x, path)
    plt.xlabel("Steps")
    plt.ylabel("Value")
    plt.title(f"Random walk steps : {len(path)}")
    plt.show()
    plt.savefig("misc/random_walk.jpg", dpi=300)

    print("Saved Fig")
path = elementary_random_walk()
plot_random_walk(path)