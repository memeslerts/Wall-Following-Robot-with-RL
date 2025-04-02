#!/usr/bin/python3
import rospy
import numpy as np
from env import Environment
import os

def q_learning():
    print("begin training...")
    env = Environment()


    q_table = {
        ("ok","ok"): { #turn left
            "forward": 0.0,
            "left": 0.0,
            "right":0.0,
            "sharp left":0.0,
            "sharp right": 0.0
        },
        ("far", "ok"): { #keep moving forward
            "forward": 0.0,
            "left": 0.0,
            "right": 0.0,
            "sharp left":0.0,
            "sharp right": 0.0
        },
        ("ok", "far"): { #keep moving forward
            "forward": 0.0,
            "left": 0.0,
            "right": 0.0,
            "sharp left":0.0,
            "sharp right": 0.0
        },
        ("far", "too close"): { #too close to right wall, turn left
            "forward": 0.0,
            "left": 0.0,
            "right": 0.0,
            "sharp left":0.0,
            "sharp right": 0.0
        },
        ("far", "far"): {
            "forward": 0.0,
            "left": 0.0,
            "right": 0.0,
            "sharp left":0.0,
            "sharp right": 0.0
        },
        ("too close", "ok"): {# front wall collision
            "forward": 0.0,
            "left": 0.0,
            "right": 0.0,
            "sharp left":0.0,
            "sharp right": 0.0
        },
        ("too close", "far"): {# front wall collision
            "forward": 0.0,
            "left": 0.0,
            "right": 0.0,
            "sharp left":0.0,
            "sharp right": 0.0
        },
        ("ok", "too far"): { #time to turn but not following wall
            "forward": 0.0,
            "left": 0.0,
            "right": 0.0,
            "sharp left": 0.0,
            "sharp right": 0.0
        },
        ("ok", "too close"): { #too close to right wall, move away from right wall
            "forward": 0.0,
            "left": 0.0,
            "right": 0.0,
            "sharp left":0.0,
            "sharp right": 0.0
        },
        ("too close", "too far"): { #too close to front wall, too far right wall
            "forward": 0.0,
            "left": 0.0,
            "right": 0.0,
            "sharp left":0.0,
            "sharp right": 0.0
        },
        ("too close", "too close"): { #too close to front wall, too far right wall
            "forward": 0.0,
            "left": 0.0,
            "right": 0.0,
            "sharp left":0.0,
            "sharp right": 0.0
        },
        ("far", "too far"): { #move forward, too far right wall 
            "forward": 0.0,
            "left": 0.0,
            "right": 0.0,
            "sharp left":0.0,
            "sharp right": 0.0
        },
        ("too close", "close"): {
            "forward": 0.0,
            "left": 0.0,
            "right": 0.0,
            "sharp left":0.0,
            "sharp right": 0.0
        },
        ("far", "close"): {
            "forward": 0.0,
            "left": 0.0,
            "right": 0.0,
            "sharp left":0.0,
            "sharp right": 0.0
        },
        ("ok", "close"): {
            "forward": 0.0,
            "left": 0.0,
            "right": 0.0,
            "sharp left":0.0,
            "sharp right": 0.0
        }
    }

    alpha = 0.2 #learning rate
    gamma = 0.8 #discount factor
    epsilon = 0.92 #exploration
    epochs = 1000
    avg_rewards = []
    total_rewards = []
    avg_errors = []

    for epoch in range(epochs):
        print("Running epoch {}".format(epoch))
        state = env.reset()
        done = False
        rewards = []
        errors = []

        while not done:
            action = env.policy(state, q_table, epsilon)

            next_state, reward, done = env.move(action)

            state_dis = state
            next_state_dis = next_state

            current_q_value = q_table[state_dis].get(action, 0.0)
            next_q_value = max(q_table[next_state_dis].values(), default=0.0)
            target = reward + gamma * next_q_value
            err = target - current_q_value

            q_table[state_dis[0], state_dis[1]][action] += alpha * err

            state = next_state

            rewards.append(reward)
            errors.append(err)
        avg_rewards.append(np.mean(rewards))
        total_rewards.append(np.sum(rewards))
        avg_errors.append(np.mean(errors))

        epsilon = max(epsilon * 0.99, 0.1)
        np.save("/home/mimilertsaroj/catkin_ws/src/wall_follower/scripts/q_table_ql.npy", q_table)
        np.save("/home/mimilertsaroj/catkin_ws/src/wall_follower/scripts/rewards_ql.npy", avg_rewards)
        np.save("/home/mimilertsaroj/catkin_ws/src/wall_follower/scripts/errors_ql.npy", avg_errors)
        np.save("/home/mimilertsaroj/catkin_ws/src/wall_follower/scripts/rewards_total_ql.npy", total_rewards)
    # Save the q_table
    print(q_table)
    print("saving...")



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
    
    q_table_path = "/home/mimilertsaroj/catkin_ws/src/wall_follower/scripts/q_table_ql.npy"
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
    # q_learning()
    q_learning_test()
