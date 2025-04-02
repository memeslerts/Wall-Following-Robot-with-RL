#!/usr/bin/python3
import rospy
import numpy as np
from env import Environment
import os

def sarsa():
    print("begin training...")
    env = Environment()

    q_table = {
        ("ok", "ok"): {
            "forward": 0.0,
            "left": 0.0,
            "right": 0.0,
            "sharp left": 0.0,
            "sharp right": 0.0
        },
        ("far", "ok"): {
            "forward": 0.0,
            "left": 0.0,
            "right": 0.0,
            "sharp left": 0.0,
            "sharp right": 0.0
        },
        ("ok", "far"): {
            "forward": 0.0,
            "left": 0.0,
            "right": 0.0,
            "sharp left": 0.0,
            "sharp right": 0.0
        },
        ("far", "too close"): {
            "forward": 0.0,
            "left": 0.0,
            "right": 0.0,
            "sharp left": 0.0,
            "sharp right": 0.0
        },
        ("far", "far"): {
            "forward": 0.0,
            "left": 0.0,
            "right": 0.0,
            "sharp left": 0.0,
            "sharp right": 0.0
        },
        ("too close", "ok"): {
            "forward": 0.0,
            "left": 0.0,
            "right": 0.0,
            "sharp left": 0.0,
            "sharp right": 0.0
        },
        ("too close", "far"): {
            "forward": 0.0,
            "left": 0.0,
            "right": 0.0,
            "sharp left": 0.0,
            "sharp right": 0.0
        },
        ("ok", "too far"): {
            "forward": 0.0,
            "left": 0.0,
            "right": 0.0,
            "sharp left": 0.0,
            "sharp right": 0.0
        },
        ("ok", "too close"): {
            "forward": 0.0,
            "left": 0.0,
            "right": 0.0,
            "sharp left": 0.0,
            "sharp right": 0.0
        },
        ("too close", "too far"): {
            "forward": 0.0,
            "left": 0.0,
            "right": 0.0,
            "sharp left": 0.0,
            "sharp right": 0.0
        },
        ("too close", "too close"): {
            "forward": 0.0,
            "left": 0.0,
            "right": 0.0,
            "sharp left": 0.0,
            "sharp right": 0.0
        },
        ("far", "too far"): {
            "forward": 0.0,
            "left": 0.0,
            "right": 0.0,
            "sharp left": 0.0,
            "sharp right": 0.0
        },
        ("too far", "too far"): {
            "forward": 0.0,
            "left": 0.0,
            "right": 0.0,
            "sharp left": 0.0,
            "sharp right": 0.0
        },
        ("too close", "close"): {
            "forward": 0.0,
            "left": 0.0,
            "right": 0.0,
            "sharp left": 0.0,
            "sharp right": 0.0
        },
        ("far", "close"): {
            "forward": 0.0,
            "left": 0.0,
            "right": 0.0,
            "sharp left": 0.0,
            "sharp right": 0.0
        },
        ("ok", "close"): {
            "forward": 0.0,
            "left": 0.0,
            "right": 0.0,
            "sharp left": 0.0,
            "sharp right": 0.0
        },
    }

    alpha = 0.3 #learning rate
    gamma = 0.8 #discount factor
    epsilon = 0.92 #exploration
    epochs = 1000
    avg_rewards = []
    avg_errors = []
    total_rewards =[]

    for epoch in range(epochs):
        print("Running epoch {}".format(epoch))
        state = env.reset()
        action = env.policy(state, q_table, epsilon)
        done = False
        rewards = []
        errs = []
        while not done:
            next_state, reward, done = env.move(action)
            next_action = env.policy(next_state, q_table, epsilon)

            state_dis = state
            next_state_dis = next_state

            current_q_value = q_table[state_dis].get(action, 0.0)
            next_q_value = q_table[next_state_dis].get(next_action, 0.0)
            target = reward + gamma * next_q_value
            err = target - current_q_value

            q_table[state_dis][action] += alpha * err

            state = next_state
            action = next_action
            rewards.append(reward)
            # print(reward)
            errs.append(err)
        # print(rewards)
        avg_rewards.append(np.mean(rewards))
        print(f"average reward for this epoch: {np.mean(rewards)}")
        total_rewards.append(np.sum(rewards))
        avg_errors.append(np.mean(errs))
        print(f"average error for this epoch: {np.mean(errs)}")

        epsilon = max(epsilon * 0.99, 0.1)

    print(q_table)
    print("Saving...")
    np.save("/home/mimilertsaroj/catkin_ws/src/wall_follower/scripts/q_table_sarsa.npy", q_table)
    np.save("/home/mimilertsaroj/catkin_ws/src/wall_follower/scripts/sarsa_errors.npy", avg_errors)
    np.save("/home/mimilertsaroj/catkin_ws/src/wall_follower/scripts/sarsa_rewards.npy", avg_rewards)
    np.save("/home/mimilertsaroj/catkin_ws/src/wall_follower/scripts/sarsa_rewards_total.npy", total_rewards)

def load_q_table(file_path):
    """Load Q-table from file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Q-table file not found at {file_path}")
    
    q_table = np.load(file_path, allow_pickle=True).item()
    print("Q-table loaded successfully")
    return q_table

def sarsa_test():
    print("Starting Q-learning test...")
    env = Environment()
    
    q_table_path = "/home/mimilertsaroj/catkin_ws/src/wall_follower/scripts/q_table_sarsa_n.npy"
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
    # sarsa()
    sarsa_test()
