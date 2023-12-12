# vision-based-robotic-grasping

CS5180 Project -- Northeastern University

Submitted by - Sarvesh Prajapati and Yash Mewada

Implementation of Deep Learning for Vision-Based Robotic Grasping



## For Training

```
# Stablebaselines -- DDPG and A2C
python3 vision-based-robotic-grasping/vision_based_rl_grasping/scripts/vision-paper-replication/trainer/main.py
# Scratcch -- DQN and PPO
python3 vision-based-robotic-grasping/vision_based_rl_grasping/scripts/vision-paper-replication/trainer/trainer.py

```

## For Sim2Real

```
# Install ROS Noetic
# Install IsaacSim 2022.2.1
# Install Moveit
catkin build
source devel/setup.bash
roslaunch vision-based-robotic-grasping ur3e_robotiq_moveit_config demo.launch
# Launch Isaac Sim and load in the world with DQN script. (Ensure ROS Bridge is working).
```

## To-Do

* Add ArgParser for training different models
* Parallelize Environment

### Thanks
1. OpenAI Gym
2. OpenAI SpinUp
3. PyBullets
4. StableBaselines_3
5. Isaac Orbit
6. ROS
7. CS5180 -- EX6
