#!/usr/bin/python3
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Pose
from std_srvs.srv import Empty
from gazebo_msgs.srv import GetModelState, SetModelState
from gazebo_msgs.msg import ModelState
import time
import tf.transformations

class Environment:
    def __init__(self):
        print("Initializing environment...")
        rospy.init_node('environment')
        rospy.set_param('/use_sim_time', True)
        time.sleep(1)
        
        self.sub = rospy.Subscriber("/scan", LaserScan, self.lidar_callback)
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.lidar_data = None
        self.reset_sim = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.step_count = 0
        self.consecutive_steps_without_movement = 0
        self.last_position = None
        self.successful_steps = 0
        self.max_steps_per_episode = 10000
        self.success_threshold = 1500
        
        self.movement_threshold = 0.01  #meters
        self.stuck_threshold = 10  #consecutive steps

    def lidar_callback(self, data):
        self.lidar_data = data.ranges

    def discretise(self, vals):
        front_distance = float(vals[0])
        right_distance = float(vals[1])
        
        front_state = "too close" if front_distance < 0.20000001 else "ok" if front_distance <= 0.7 else "far"
        right_state = "close" if right_distance < 0.23 else "too close" if right_distance < 0.20000001 else "ok" if (right_distance <= 0.5 and right_distance >=0.345) else "too far" if right_distance > 1.0 else "far"
        
        return front_state, right_state

    def get_state(self):
        if self.lidar_data is None:
            return None
            
        front_distance = min(min(self.lidar_data[0:40]), min(self.lidar_data[320:360]))
        right_distance = min(self.lidar_data[210:340])
        
        state = self.discretise((front_distance, right_distance))
        return state
    
    def get_current_position(self):
        try:
            model_state = self.get_model_state('triton', '')
            return model_state.pose.position
        except rospy.ServiceException as e:
            rospy.logerr("service call failed: %s" % e)
            return None

    def has_moved(self, current_position):
        if self.last_position is None:
            self.last_position = current_position
            return True
            
        dx = abs(current_position.x - self.last_position.x)
        dy = abs(current_position.y - self.last_position.y)
        moved = (dx > self.movement_threshold) or (dy > self.movement_threshold)
        
        self.last_position = current_position
        return moved

    def policy(self, state, q_table, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(["forward", "left", "right", "sharp left", "sharp right"])
        else:
            front_state, right_state = state
            action_values = q_table.get((front_state, right_state), {})
            if not action_values:
                return np.random.choice(["forward", "left", "right", "sharp left", "sharp right"])
            return max(action_values, key=action_values.get)

    def reset(self):
        rospy.loginfo("resetting simulation...")
        
        #reset simulation
        self.reset_sim()
        rospy.set_param('/use_sim_time', True)
        
        self.step_count = 0
        self.consecutive_steps_without_movement = 0
        self.successful_steps = 0
        self.last_position = None
        
        is_random = np.random.choice([True,False])
        if True: 
            self.set_random_position()
        
        time.sleep(2) 
        return self.get_state()

    def set_random_position(self):
        try:
            state_msg = ModelState()
            state_msg.model_name = 'triton'
            
            # Set random position (adjust ranges as needed)
            # state_msg.pose.position.x = 2.0
            # state_msg.pose.position.y = -1.5
            state_msg.pose.position.x = np.random.choice([-3.5,-2.0])
            state_msg.pose.position.y = np.random.choice([0.5,2.5,-3.5])
            state_msg.pose.position.z = 0.0
            
            # Set random orientation
            # yaw = np.random.uniform(0, 2*np.pi)
            # quat = tf.transformations.quaternion_from_euler(0, 0, yaw)
            # state_msg.pose.orientation.x = quat[0]
            # state_msg.pose.orientation.y = quat[1]
            # state_msg.pose.orientation.z = quat[2]
            # state_msg.pose.orientation.w = quat[3]
            
            self.set_model_state(state_msg)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)

    def get_reward(self, state, action, next_state):
        if state is None or next_state is None:
            return 0

        cur_dist = next_state[1]
        reward = 0

        if cur_dist == "ok":
            reward = 20.0
            self.successful_steps += 1
        elif cur_dist == "far":
            reward = -10.0
        elif cur_dist == "close":
            reward = -10.0
            if action == "left":
                reward += 6.0
        elif cur_dist == "too close":
            reward = -20.0
        elif cur_dist == "too far":
            reward = -20.0

        if next_state[0] == "ok" or next_state[0] == "close":
            if action == "sharp left":
                reward += 7
        if next_state[0] == "far":
            if cur_dist == "far":
                if action == "sharp right":
                    reward += 7.5
            elif cur_dist == "ok" and action == "forward":
                reward += 7.5

        if "too close" in next_state: #collision
            reward = -40.0
        if "too far" in next_state: #deviation
            reward = -40.0

        reward += 0.25 

        return reward

    def move(self, action):
        self.step_count += 1
        
        cur_state = self.get_state()
        
        cmd_vel = Twist()
        
        if action == "forward":
            cmd_vel.linear.x = 0.3
            rospy.logdebug("moving forward")
        elif action == "left":
            cmd_vel.linear.x = 0.25
            cmd_vel.angular.z = 0.25
            rospy.logdebug("turning left")
        elif action == "right":
            cmd_vel.linear.x = 0.25
            cmd_vel.angular.z = -0.25
            rospy.logdebug("turning right")
        elif action == "sharp left":
            cmd_vel.linear.x = 0.225
            cmd_vel.angular.z = 0.75
            rospy.logdebug("sharp left turn")
        elif action == "sharp right":
            cmd_vel.linear.x = 0.225
            cmd_vel.angular.z = -0.75
            rospy.logdebug("sharp right turn")

        self.pub.publish(cmd_vel)
        time.sleep(0.1)
        
        next_state = self.get_state()
        current_position = self.get_current_position()
        if current_position is not None:
            if not self.has_moved(current_position):
                self.consecutive_steps_without_movement += 1
            else:
                self.consecutive_steps_without_movement = 0
        
        reward = self.get_reward(cur_state, action, next_state)
        rospy.logdebug(f"Current reward: {reward}")
        
        done = self.is_done(next_state)
        
        return next_state, reward, done

    def is_done(self, state):
        #check for collision
        if state is None or state[0] == "too close" or state[1] == "too close":
            rospy.loginfo("Terminating episode: collision detected")
            return True
            
        #check if robot is too far from wall
        if state[1] == "too far":
            rospy.loginfo("Terminating episode: too far from wall")
            return True
            
        #check if robot is stuck
        if self.consecutive_steps_without_movement >= self.stuck_threshold:
            rospy.loginfo(f"Terminating episode: robot stuck for {self.consecutive_steps_without_movement} steps")
            return True
            
        #check if successful policy is learned
        if self.successful_steps >= self.success_threshold:
            rospy.loginfo(f"Terminating episode: successful policy achieved for {self.successful_steps} steps")
            return True
            
        #check if max steps reached
        if self.step_count >= self.max_steps_per_episode:
            rospy.loginfo(f"Terminating episode: max steps ({self.max_steps_per_episode}) reached")
            return True
            
        return False