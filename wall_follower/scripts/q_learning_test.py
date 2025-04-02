#!/usr/bin/python3
import numpy as np
from env import Environment
import os
import rospy


def load_q_table(file_path):
    """Load Q-table from file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Q-table file not found at {file_path}")
    
    q_table = np.load(file_path, allow_pickle=True).item()
    print("Q-table loaded successfully")
    return q_table

def q_learning_test():
    print("Starting Q-learning test...")
    env = Environment()
    
    q_table_path = "/home/mimilertsaroj/catkin_ws/src/wall_follower/scripts/q_table_ql_n.npy"
    try:
        q_table = load_q_table(q_table_path)
    except Exception as e:
        print(f"Error loading Q-table: {e}")
        return
    
    epsilon = 0.0  
    test_episodes = 5
    
    for episode in range(test_episodes):
        print(f"\n=== Testing Episode {episode + 1}/{test_episodes} ===")
        state = env.reset()
        done = False
        step_count = 0
        
        while not done:
            action = env.policy(state, q_table, epsilon)
            next_state, reward, done = env.move(action)
            
            print(f"Step {step_count}: State={state}, Action={action}, Reward={reward:.2f}")
            
            state = next_state
            step_count += 1
            
            rospy.sleep(0.1)
        
        print(f"episode {episode + 1} completed in {step_count} steps")
    
    print("\ncompleted!")

if __name__ == "__main__":
    q_learning_test()