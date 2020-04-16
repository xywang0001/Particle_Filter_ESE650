#*
#    SLAM.py: the implementation of SLAM
#    Feb 2020
#*
import numpy as np
import matplotlib.pyplot as plt
import load_data as ld
import os, sys, time
import p3_util as ut
from read_data import LIDAR, JOINTS
import probs_utils as prob
import math
import cv2
import transformations as tf
from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D
sys.path.insert(0, 'MapUtils')
from bresenham2D import bresenham2D
import logging
if (sys.version_info > (3, 0)):
    import pickle
else:
    import cPickle as pickle

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))
 

class SLAM(object):
    def __init__(self):
        self._characterize_sensor_specs()
    
    def _read_data(self, src_dir, dataset=0, split_name='train'):
        self.dataset_= str(dataset)
        if split_name.lower() not in src_dir:
            src_dir  = src_dir + '/' + split_name
        print('\n------Reading Lidar and Joints (IMU)------')
        self.lidar_  = LIDAR(dataset=self.dataset_, data_folder=src_dir, name=split_name + '_lidar'+ self.dataset_)
        print ('\n------Reading Joints Data------')
        self.joints_ = JOINTS(dataset=self.dataset_, data_folder=src_dir, name=split_name + '_joint'+ self.dataset_)

        self.num_data_ = len(self.lidar_.data_)
        # Position of odometry
        self.odo_indices_ = np.empty((2,self.num_data_),dtype=np.int64)

    def _characterize_sensor_specs(self, p_thresh=None):
        # High of the lidar from the ground (meters)
        self.h_lidar_ = 0.93 + 0.33 + 0.15
        # Accuracy of the lidar
        self.p_true_ = 9
        self.p_false_ = 1.0/9
        
        #TODO: set a threshold value of probability to consider a map's cell occupied  
        self.p_thresh_ = 0.6 if p_thresh is None else p_thresh # > p_thresh => occupied and vice versa
        # Compute the corresponding threshold value of logodd
        self.logodd_thresh_ = prob.log_thresh_from_pdf_thresh(self.p_thresh_)
        

    def _init_particles(self, num_p=0, mov_cov=None, particles=None, weights=None, percent_eff_p_thresh=None):
        # Particles representation
        self.num_p_ = num_p
        #self.percent_eff_p_thresh_ = percent_eff_p_thresh
        self.particles_ = np.zeros((3,self.num_p_),dtype=np.float64) if particles is None else particles
        
        # Weights for particles
        self.weights_ = 1.0/self.num_p_*np.ones(self.num_p_) if weights is None else weights

        # Position of the best particle after update on the map
        self.best_p_indices_ = np.empty((2,self.num_data_),dtype=np.int64)
        # Best particles
        self.best_p_ = np.empty((3,self.num_data_))
        # Corresponding time stamps of best particles
        self.time_ =  np.empty(self.num_data_)
       
        # Covariance matrix of the movement model
        tiny_mov_cov   = np.array([[1e-8, 0, 0],[0, 1e-8, 0],[0, 0 , 1e-8]])
        self.mov_cov_  = mov_cov if mov_cov is not None else tiny_mov_cov
        # To generate random noise: x, y, z = np.random.multivariate_normal(np.zeros(3), mov_cov, 1).T
        # this return [x], [y], [z]

        # Threshold for resampling the particles
        self.percent_eff_p_thresh_ = percent_eff_p_thresh
        

    def _init_map(self, map_resolution=0.05):
        '''*Input: resolution of the map - distance between two grid cells (meters)'''
        # Map representation
        MAP= {}
        MAP['res']   = map_resolution #meters
        MAP['xmin']  = -20  #meters
        MAP['ymin']  = -20
        MAP['xmax']  =  20
        MAP['ymax']  =  20
        MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
        MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))

        MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8
        self.MAP_ = MAP

        self.log_odds_ = np.zeros((self.MAP_['sizex'],self.MAP_['sizey']),dtype = np.float64)
        self.occu_ = np.ones((self.MAP_['sizex'],self.MAP_['sizey']),dtype = np.float64)
        # Number of measurements for each cell
        self.num_m_per_cell_ = np.zeros((self.MAP_['sizex'],self.MAP_['sizey']),dtype = np.uint64)


    def _build_first_map(self,t0=0,use_lidar_yaw=True):
        """Build the first map using first lidar"""
        self.t0 = t0
        # Extract a ray from lidar data
        MAP = self.MAP_
        print('\n--------Doing build the first map--------')
            

        #TODO: student's input from here 
        
        ##Get data from Lidar data, joint data & time sync
        
        ranges = self.lidar_.data_[self.t0]["scan"][0]
        lidar_t = self.lidar_.data_[self.t0]["t"]
        lidar_pose = self.lidar_.data_[self.t0]["pose"][0]
        lidar_yaw = self.lidar_.data_[self.t0]["rpy"][0][2]
        joint_ind = np.abs(self.joints_.data_["ts"][0] - lidar_t).argmin() #time sync
        angles = np.array([np.arange(-135,135.25,0.25)*np.pi/180.])
        
        #limit scan range
        indv_range = np.logical_and((ranges < 30),(ranges> 0.1))
        ranges = ranges[indv_range]
        angles = angles[0]
        angles = angles[indv_range]
        
        #x,y positions in lidar frame
        
        x_lidar = ranges * np.cos(angles)
        y_lidar = ranges * np.sin(angles)
        length = len(x_lidar)
        z_lidar = np.zeros(length)
        w_lidar = np.ones(length)
        p_lidar = np.vstack([x_lidar, y_lidar, z_lidar, w_lidar])
        
        neck_angle = self.joints_.data_["head_angles"][0][joint_ind]
        head_angle = self.joints_.data_["head_angles"][1][joint_ind]
        
        ## lidar2body & body2world
        
        #lidar2body
        body_2_lidar_rot = np.dot(tf.rot_z_axis(neck_angle), tf.rot_y_axis(head_angle))
        # Transition from the body to the lidar frame (lidar is 15cm above the head. See config_slam.pdf for more details)
        body_2_lidar_trans = np.array([0,0,0.15])
        body_2_lidar_homo = tf.homo_transform(body_2_lidar_rot,body_2_lidar_trans)
        p_body = np.dot(body_2_lidar_homo, p_lidar)
        
        #body2world
        world_2_body_rot = tf.rot_z_axis(lidar_yaw)
        world_2_body_trans = np.array([lidar_pose[0], lidar_pose[1], 0.93])
        world_2_part_homo = tf.homo_transform(world_2_body_rot,world_2_body_trans)
        p_world = np.dot(world_2_part_homo, p_body)
       
        #ground removal
        ground_ind = np.argwhere(p_world[2] < 0.1)
        p_world = np.delete(p_world,ground_ind,1)
        p_world = p_world[0:2,:]
        
        #map into map index
        sx = np.ceil((lidar_pose[0] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
        sy = np.ceil((lidar_pose[1] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
        Ex = np.ceil((p_world[0,:] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
        Ey = np.ceil((p_world[1,:] - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
        
        #brehensam 2D
        num = len(Ex)
        for i in range(num):
            r = bresenham2D(sx,sy,Ex[i], Ey[i])
            r = r.astype(np.int16)
            self.log_odds_[r[0], r[1]] += np.log(self.p_false_)
            
        self.log_odds_[Ex, Ey] += (np.log(self.p_true_) - np.log(self.p_false_))
        
        MAP["map"] = MAP["map"].astype(np.float64)
        MAP["map"] += self.log_odds_
        
        MAP["map"] = 1.0 - 1.0/(1.0 + np.exp(MAP["map"]))
        
        obst = MAP["map"] > self.p_thresh_
        free = MAP["map"] < 0.2
        unexp = (MAP["map"]> 0.2) & (MAP["map"] < self.p_thresh_)
               
        MAP["map"][obst] = 0
        MAP["map"][free] = 1
        MAP["map"][unexp] = 0.5     

        #End student's input 

        self.MAP_ = MAP

    def _predict(self,t,use_lidar_yaw=True):
        
        logging.debug('\n-------- Doing prediction at t = {0}------'.format(t))
        #TODO: student's input from here
        lidar_pose_t = self.lidar_.data_[t]["pose"][0]
        lidar_pose_prev = self.lidar_.data_[t-1]["pose"][0]
        yaw_prev = self.lidar_.data_[t-1]["rpy"][0][2]
        yaw_current = self.lidar_.data_[t]["rpy"][0][2]
        delta_yaw = yaw_current - yaw_prev
        #yaw_prev = lidar_pose_prev[2]     
        T_prev = np.array([[np.cos(yaw_prev), np.sin(yaw_prev),0],[-np.sin(yaw_prev), np.cos(yaw_prev),0],[0,0,1]])
        
        d_pose = lidar_pose_t - lidar_pose_prev
        d_pose = np.array([d_pose[0], d_pose[1], delta_yaw])
        d_pose = np.dot(T_prev, d_pose)
        
        for i in range(self.num_p_):
            yaw_est = self.particles_[2,i]
            R = np.array([[np.cos(yaw_est), -np.sin(yaw_est),0],[np.sin(yaw_est), np.cos(yaw_est),0],[0,0,1]])
            d_pose_est = np.dot(R, d_pose)
            x,y,z = np.random.multivariate_normal(np.zeros(3), self.mov_cov_ , 1).T
            self.particles_[0,i] += x + d_pose_est[0] 
            self.particles_[1,i] += y + d_pose_est[1] 
            self.particles_[2,i] += z + d_pose_est[2] 
       
        #End student's input 

    def _update(self,t,t0=0,fig='on'):
        """Update function where we update the """
        if t == t0:
            self._build_first_map(t0,use_lidar_yaw=True)
            return

        #TODO: student's input from here 
        
        n_thresh = 5
        MAP = self.MAP_
        #trajectory update
        ranges = self.lidar_.data_[t]["scan"][0]
        lidar_t = self.lidar_.data_[t]["t"]
        lidar_pose = self.lidar_.data_[t]["pose"][0]
        #lidar_yaw = self.lidar_.data_[self.t0]["rpy"][0][2]
        joint_ind = np.abs(self.joints_.data_["ts"][0] - lidar_t).argmin() #time sync
        angles = np.array([np.arange(-135,135.25,0.25)*np.pi/180.])
        
        #limit scan range
        indv_range = np.logical_and((ranges < 30),(ranges> 0.1))
        ranges = ranges[indv_range]
        angles = angles[0]
        angles = angles[indv_range]
        
        x_lidar = ranges * np.cos(angles)
        y_lidar = ranges * np.sin(angles)
        length = len(x_lidar)
        z_lidar = np.zeros(length)
        w_lidar = np.ones(length)
        p_lidar = np.vstack([x_lidar, y_lidar, z_lidar, w_lidar])
        
        neck_angle = self.joints_.data_["head_angles"][0][joint_ind]
        head_angle = self.joints_.data_["head_angles"][1][joint_ind]
        
        #lidar2body
        body_2_lidar_rot = np.dot(tf.rot_z_axis(neck_angle), tf.rot_y_axis(head_angle))
        # Transition from the body to the lidar frame (lidar is 15cm above the head. See config_slam.pdf for more details)
        body_2_lidar_trans = np.array([0,0,0.15])
        body_2_lidar_homo = tf.homo_transform(body_2_lidar_rot,body_2_lidar_trans)
        p_body = np.dot(body_2_lidar_homo, p_lidar)
        
        corr = np.zeros(self.num_p_)
        
        for i in range(self.num_p_):
            
            pose_i = self.particles_[:,i]
            world_2_body_rot = tf.rot_z_axis(pose_i[2])
            world_2_body_trans = np.array([pose_i[0], pose_i[1], 0.93])
            world_2_part_homo = tf.homo_transform(world_2_body_rot,world_2_body_trans)
            p_world_est = np.dot(world_2_part_homo, p_body)
            
            ground_ind = np.argwhere(p_world_est[2] < 0.1)
            p_world_est = np.delete(p_world_est,ground_ind,1)
            p_world_est = p_world_est[0:2,:]
            
            Ex = np.ceil((p_world_est[0,:] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
            Ey = np.ceil((p_world_est[1,:] - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
            
            obst = np.vstack([Ex,Ey])
            
            corr[i] = prob.mapCorrelation(MAP["map"],obst)
        
        #update weights
        
        self.weights_ = prob.update_weights(self.weights_ , corr)
        max_ = np.argmax(self.weights_)
        max_pose = self.particles_[:,max_]
        self.best_p_[:,t] = max_pose
        
        ind_x = np.ceil((max_pose[0] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
        ind_y = np.ceil((max_pose[1] - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
        
        self.best_p_indices_[0,t] = ind_x
        self.best_p_indices_[1,t] = ind_y
        
        if t == 1:
            
            self.best_p_indices_[0,t-1] = ind_x
            self.best_p_indices_[1,t-1] = ind_y
            
        
        n_eff = 1.0/(np.sum(self.weights_)**2)
        
        
        if n_eff < self.percent_eff_p_thresh_ * self.num_p_:
            self.particles_, self.weights_ = prob.stratified_resampling(self.particles_,self.weights_,self.num_p_)   

        #self.time_ =  np.empty(self.num_data_)          
            
        #map_update

        world_2_body_rot = tf.rot_z_axis(max_pose[2])
        world_2_body_trans = np.array([max_pose[0], max_pose[1], 0.93])
        world_2_part_homo = tf.homo_transform(world_2_body_rot,world_2_body_trans)
        p_world_est = np.dot(world_2_part_homo, p_body)

        ground_ind = np.argwhere(p_world_est[2] < 0.1)
        p_world_est = np.delete(p_world_est,ground_ind,1)
        p_world_est = p_world_est[0:2,:]

        #Ex = np.ceil((p_world_est[0,:] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
        #Ey = np.ceil((p_world_est[1,:] - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1

        #map into map index
        sx = np.ceil((max_pose[0] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
        sy = np.ceil((max_pose[1] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
        Ex = np.ceil((p_world_est[0,:] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
        Ey = np.ceil((p_world_est[1,:] - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
        
        #brehensam 2D
        num = len(Ex)
        for i in range(num):
            r = bresenham2D(sx,sy,Ex[i], Ey[i])
            r = r.astype(np.int16)
            self.log_odds_[r[0], r[1]] += np.log(self.p_false_)
            
        self.log_odds_[Ex, Ey] += (np.log(self.p_true_) - np.log(self.p_false_))
        
        MAP["map"] = MAP["map"].astype(np.float64)
        MAP["map"] += self.log_odds_
        
        MAP["map"] = 1.0 - 1.0/(1.0 + np.exp(MAP["map"]))
        
        obst = MAP["map"] > self.p_thresh_
        free = MAP["map"] < 0.2
        unexp = (MAP["map"]> 0.2) & (MAP["map"] < self.p_thresh_)
               
        MAP["map"][obst] = 0
        MAP["map"][free] = 1
        MAP["map"][unexp] = 0.5

        #End student's input 

        self.MAP_ = MAP
        return self.MAP_ 
