# SLAM_ESE650

## Problem Statement
Implement the structure of mapping and localization in an indoor environment with particle filters, using information from mainly from IMU and LIDAR. Particle filter is a non-parametric implementation of the Bayes filter. With the particle filter, the posterior bel(xt) is represented by a set of random state samples drawn from this posterior.

## User's Guide
1. The main SLAM structure is implemented in <div>SLAM.py</div>
2. A set of parameters should be changed in <div>main.py</div>, including map resolutions, noise level and 
3. Set the dataset and run <div>python main.py</div>.

## Result
1. Dataset 0 
![image](https://github.com/xywang0001/Particle_Filter_ESE650/blob/master/results/processing_SLAM_map_train_0.jpg)

2. Dataset 2
![image](https://github.com/xywang0001/Particle_Filter_ESE650/blob/master/results/processing_SLAM_map_train_2.jpg)

3. Dataset 3
![image](https://github.com/xywang0001/Particle_Filter_ESE650/blob/master/results/processing_SLAM_map_train_3.jpg)
