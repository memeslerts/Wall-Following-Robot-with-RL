Requirements
- ROS Noetic
- Gazebo
- Python 3
- stingray_sim package
- numpy

This package contains:
- Code for training SARSA and Q Learning
- Code for testing SARSA and Q Learning
- Q Tables for SARSA and Q Learning

Installation and Loading the WS
1. Install ROS Noetic 
2. Create a Catkin Workspace
3. Clone the Repository for stingray_sim
    cd ~/catkin_ws/src
    git clone https://gitlab.com/HCRLab/stingray-robotics/stingray_sim.git
4. Build the workspace
    cd ~/catkin_ws
    catkin_make
    source devel/setup.bash'

Running P2D1 (Wall Following Manual Code):
1. Run roslauch wall_follower wall_following.launch

Running Any Script on P2D2:
1. In a new terminal run
    cd catkin_ws
    source devel/setup.bash
2. For Q Learning:
    Training: roslaunch wall_follower q_learning_train.launch
    Testing: roslaunch wall_follower q_learning_test.launch
3. For SARSA:
    Training: roslaunch wall_follower sarsa_train.launch
    Testing: roslaunch wall_follower sarsa_test.launch