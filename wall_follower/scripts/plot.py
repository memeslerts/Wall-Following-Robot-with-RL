#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import os

def load_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"file not found at {file_path}")
    
    arr = np.load(file_path)
    print("array loaded successfully")
    return arr

def plot_arr(data):
    plt.plot(data)
    plt.show()

if __name__ == "__main__":
    array = load_file('/home/mimilertsaroj/catkin_ws/src/wall_follower/scripts/sarsa_errors.npy')
    plot_arr(array)
