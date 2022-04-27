import numpy as np
#import cv2
from os import listdir
from os.path import isfile, join
import gym
from gym import error, spaces, utils
import glob
from PIL import Image
import open3d as o3d
from .cl import *
from skimage.morphology import binary_dilation
from .proc3d import *
import json
from .utils import *
import glob
import os
from .space_carving import *
import random

MODELS_PATH = '/home/pico/uni/romi/scanner_cube/arabidopsis_im_bigger/ramdisk'


class ScannerEnv(gym.Env):
    """
    Custom OpenAI Gym environment  for training 3d scannner
    """
    metadata = {'render.modes': ['human']}
    def __init__(self,models_path,train_models,gt_mode=True):
        super(ScannerEnv, self).__init__()
        #self.__version__ = "7.0.1"
        # if gt_mode true, ground truth model is used by space carving object (for comparing it against current volume)
        self.gt_mode = gt_mode
        # number of images that must be collected 
        self.n_images = 10 
        self.models_path = models_path
        # total of posible positions for theta  in env
        self.theta_n_positions = 180
        # total of posible positions for phi in env
        self.phi_n_positions = 4
        # 3d models used for training
        self.train_models = train_models  

        # --------------------------self.zeros_test = np.zeros((66,68,152)).astype('float16')

        '''the state returned by this environment consiste of
        the volume being carved , theta position , phi position'''
        # volume used in the carving
        self.volume_shape = (64,64,64)
        self.vol_obs_space = gym.spaces.Box(low=-1, high=1, shape=self.volume_shape, dtype=np.float16)

        # theta positions                                          
        t_low = np.array([0])
        t_high = np.array([self.theta_n_positions-1])                                           
        self.theta_obs_space = gym.spaces.Box(t_low, t_high, dtype=np.int32)

        # phi positions                                          
        p_low = np.array([0])
        p_high = np.array([self.phi_n_positions-1])                                           
        self.phi_obs_space = gym.spaces.Box(p_low, p_high, dtype=np.int32)

        '''lowl = np.array([-1]*self.n_images)
        highl = np.array([179]*self.n_images)                                           
        self.vec_ob_space = gym.spaces.Box(lowl, highl, dtype=np.int32)'''


        self.observation_space = gym.spaces.Tuple((self.vol_obs_space, self.theta_obs_space, self.phi_obs_space))


        #action space 
        self.actions = {0:1,1:5,2:10,3:15,4:20,5:25,6:30,7:35,8:40,9:45,10:50,11:55,12:60,13:65,14:70,15:75,16:80,17:85,18:90}
        self.action_space = gym.spaces.Discrete(19)

        # map action with correspondent movements in theta and phi
        # theta numbers are number of steps relative to current position
        # phi numbers are absolute position in phi
        #(theta,phi)
        self.action_map = {0:(2,0),1:(5,0),2:(10,0),3:(15,0),4:(25,0),5:(45,0),
                           6:(2,1),7:(5,1),8:(10,1),9:(15,1),10:(25,1),11:(45,1),
                           12:(2,2),13:(5,2),14:(10,2),15:(15,2),16:(25,2),17:(45,2),
                           18:(2,3),19:(5,3),20:(10,3),21:(15,3),22:(25,3),23:(45,3)}

        #self._spec.id = "Romi-v0"
        self.reset()

    def reset(self,theta_init=-1,phi_init=-1,theta_bias=-1):
        self.num_steps = 0
        self.total_reward = 0
        self.done = False

        # keep track of visited positions during the episode
        self.visited_positions = [] 

        # count of empty,undetermined and solid voxels in volume
        # -1's (empty space), 0's (undetermined) and 1's (solid) from 3d volume
        self.voxel_count = [0,0,0]
        self.last_voxel_count = [0,0,0] #last count of empty spaces (when not in gt mode)

        #inital position of the camera, if -1 choose random
        self.init_theta = theta_init
        self.init_phi = phi_init
        
        if self.init_theta == -1:
            self.init_theta = np.random.randint(0,self.theta_n_positions)
            self.current_theta = self.init_theta
        else:
            self.current_theta = self.init_theta

        if self.init_phi == -1:
            self.init_phi = np.random.randint(0,self.phi_n_positions)
            self.current_phi = self.init_phi 
        else:
            self.current_phi = self.init_phi

        # simulates rotation of object (z axis) by n steps (for data augmentation), -1 for random rotation
        if theta_bias == -1: 
            self.theta_bias = np.random.randint(0,self.theta_n_positions)
        else:
            self.theta_bias = theta_bias

        #append initial position to visited positions list    
        self.visited_positions.append((self.current_theta, self.current_phi))
    
        #---------------------------------------self.state_images[0] = self.current_position #add first image to state
        
        # take random  model from available models list
        model = random.choice(self.train_models)

        # create space carving object
        self.spc = space_carving_rotation_2d( os.path.join(self.models_path, model),
                        gt_mode=self.gt_mode, theta_bias=self.theta_bias,
                        total_theta_positions=self.theta_n_positions,
                        cube_view='static')

        # carve image from initial position
        self.spc.carve(self.current_theta, self.current_phi)
        vol = self.spc.volume
        

        if self.gt_mode is True:
            # keep similarity ratio of current volume and groundtruth volume
            # for calculating deltas of similarity ratios in next steps
            self.last_gt_ratio = self.spc.gt_compare_solid()
        else:
            #get number of -1's (empty space), 0's (undetermined) and 1's (solid) from 3d volume
            self.voxel_count = [np.count_nonzero(vol == -1),
                                np.count_nonzero(vol == 0),
                                np.count_nonzero(vol == 1) ] 
            self.last_voxel_count = self.voxel_count.copy() 
        
        self.current_state = ( vol.astype('float16') ,
                               np.array([self.current_theta],dtype=int),
                               np.array([self.current_phi],dtype=int))

        return self.current_state


  

    def step(self, action):
        self.num_steps += 1
        
        #move n steps from current position
        steps = self.actions[action]
        self.current_position = self.calculate_position(self.current_position, steps)
        
        
        
        self.visited_positions.append((self.current_theta, self.current_phi))

        #carve in new position
        self.spc.carve(self.current_theta, self.current_phi)
        vol = self.spc.volume

        
        if self.gt_mode is True:
            #calculate increment of solid voxels ratios between gt and current volume
            gt_ratio = self.spc.gt_compare_solid()
            delta_gt_ratio = gt_ratio - self.last_gt_ratio
            self.last_gt_ratio = gt_ratio
            reward = delta_gt_ratio
        
        else:
            #get number of -1's (empty space), 0's (undetermined) and 1's (solid) from 3d volume
            self.voxel_count = [np.count_nonzero(vol == -1),
                                np.count_nonzero(vol == 0),
                                np.count_nonzero(vol == 1) ] 
            #np.histogram(self.spc.sc.values(), bins=3)[0]

            ''' do some calculation with the voxel count'''
            '''#calculate increment of detected spaces since last carving
            delta = self.h[0] - self.last_vspaces_count
            reward = min(delta,30000) / 30000'''
            reward=0

            self.last_voxel_count = self.voxel_count.copy() 
        

        if self.num_steps >= (self.n_images-1):
            self.done = True
           
        self.total_reward += reward

        self.current_state = ( vol.astype('float16') ,
                               np.array([self.current_theta],dtype=int),
                               np.array([self.current_phi],dtype=int))
      

        return self.current_state, reward, self.done, {}

 
    def minMaxNorm(self,old, oldmin, oldmax , newmin , newmax):
        return ( (old-oldmin)*(newmax-newmin)/(oldmax-oldmin) ) + newmin

   
    def calculate_theta_position(self,curr_theta,steps):
        n_pos = curr_theta + steps
        if n_pos>(self.theta_n_positions-1):
            n_pos -= self.theta_n_positions
        elif n_pos<0:
            n_pos += self.theta_n_positions
        return n_pos


    @property
    def nA(self):
        return self.action_space.n

    def render(self, mode='human', close=False):
        """
        :param mode:
        :return:
        """
        return
