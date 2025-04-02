#!/usr/bin/python3
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

#following a straight wall
q_table = {
    ("far", "ok"):{ #front, left travel forward 
        "forward": 5,
        "left": 0.0,
        "right": 0.0,
        "stop": 0.0
    },
    ("ok","ok"):{ #hit the front wall
        "forward": 0.1,
        "left": 0.1,
        "right": 0.1,
        "stop": 5.0
    },
    ("far","too close"):{ #too close to left wall
        "forward":0.1,
        "left":0.1,
        "right":3,
        "stop":0.0
    },
    ("far","far"):{ #far from left wall and front wall
        "forward": 2.5,
        "left": 2.5,
        "right": 0.1,
        "stop":0.0
    },
}

def policy(state):
    if state in q_table:
        return max(q_table[state], key=q_table[state].get)
    else:
        return "stop"

def scan_callback(scan):
    ranges = scan.ranges
    # print(ranges)
    front_distance = min(min(ranges[0:20]),min(ranges[340:360]))
    left_distance = min(ranges[40:150])
    # right_distance = min(ranges[-len(ranges)//3:])
    print(front_distance)
    print(left_distance)
    state = ("too close" if  front_distance < 0.4 else "ok" if front_distance <= 0.6 else "far",
            "too close" if left_distance < 0.4 else "ok" if left_distance<=0.5 else "far")
    print(state)
    action = policy(state)
    # print(action)
    
    cmd_vel = Twist()
    move(action,cmd_vel)

def move(action,cmd_vel):
    if action == "forward":
        cmd_vel.linear.x = 0.3
        print("forward")
    elif action == "left":
        cmd_vel.linear.x = 0.05
        cmd_vel.angular.z = 0.3
        print("left")
    elif action == "right":
        cmd_vel.linear.x=0.05
        cmd_vel.angular.z=-0.3
        print("right")
    # else:
    #     cmd_vel.linear.x=0.0
    #     print("stop")
    #     rospy.signal_shutdown("done!")
    pub.publish(cmd_vel) 

def main():
    rospy.init_node("wall_follower")
    global pub
    pub = rospy.Publisher("/cmd_vel",Twist,queue_size=10)
    rospy.Subscriber("/scan",LaserScan, scan_callback)
    rospy.spin()

if __name__ == "__main__":
    main()