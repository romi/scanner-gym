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
    A template to implement custom OpenAI Gym environments
    """
    metadata = {'render.modes': ['human']}
    def __init__(self,dataset_path,init_pos_inc_rst=False,gt_mode=True,rotation_steps=0,init_pos=0):
        super(ScannerEnv, self).__init__()
        #self.__version__ = "7.0.1"
        self.init_pos = init_pos
        self.gt_mode = gt_mode
        self.rotation_steps = rotation_steps #simulates rotation of object (z axis) by n steps (for data augmentation), -1 for random rotation
        self.n_images = 20 #number of images that must be collected 
        self.dataset_path = dataset_path
        self.n_positions = 180 #total of posible positions in env
        self.init_pos_inc_rst = init_pos_inc_rst #if false init position is random, if true, it starts in position 0 and increments by 1 position every reset
        self.init_pos_counter = 0
        #self.rnd_train_models = [1,3,25,39,41,6,9,11,14,22,0,20,28,36,48,201,202,203,204,205]  #models used when using random mode (path = '')
        #self.rnd_train_models = [1,3,25,    6,9,11,   0,20,28, 201,202,203,204,205]  #models used when using random mode (path = '')
        self.rnd_train_models = [0,20,28,204,205]  #models used when using random mode (path = '')

        self.zeros_test = np.zeros((66,68,152)).astype('float16')
        
        self.volume_shape = (66,68,152)
        self.im_ob_space = gym.spaces.Box(low=-1, high=1, shape=self.volume_shape, dtype=np.float16)

        #current position                                          
        lowl = np.array([0])
        highl = np.array([179])                                           
        self.vec_ob_space = gym.spaces.Box(lowl, highl, dtype=np.int32)
        
        

        '''lowl = np.array([-1]*self.n_images)
        highl = np.array([179]*self.n_images)                                           
        self.vec_ob_space = gym.spaces.Box(lowl, highl, dtype=np.int32)'''


        self.observation_space = gym.spaces.Tuple((self.im_ob_space, self.vec_ob_space))
        #self.observation_space = self.im_ob_space



        #self.actions = {0:1,1:3,2:5,3:11,4:23,5:33,6:45,7:60,8:-60,9:-45,10:-33,11:-23,12:-11,13:-5,14:-3,15:-1,16:90}
        #self.action_space = gym.spaces.Discrete(17)
        
        self.actions = {0:1,1:5,2:10,3:15,4:20,5:25,6:30,7:35,8:40,9:45,10:50,11:55,12:60,13:65,14:70,15:75,16:80,17:85,17:90}
        self.action_space = gym.spaces.Discrete(18)

        #self._spec.id = "Romi-v0"
        self.reset()

    def reset(self):
        self.num_steps = 0
        self.total_reward = 0
        self.done = False
        self.kept_images = [] # position of images in dataset
        self.state_images = np.array([-1]*self.n_images) #used as part of state (-1 means empty)
        self.h = [0,0,0] # count of empty,undetermined and solid voxels
        self.last_vspaces_count = 0 #count of empty spaces (when not in gt mode)

        
            
        if self.init_pos_inc_rst : #initial position increases at every reset 
            if self.init_pos_counter >= self.n_positions:
                self.init_pos_counter = 0
            self.current_position = self.init_pos_counter
            self.init_pos_counter += 1
        else:
            if self.init_pos == -1: #random inital position
                self.current_position = np.random.randint(0,self.n_positions)
            else:
                self.current_position = self.init_pos
         
        if self.rotation_steps == -1: #rotation bias ( rotating plant) set randomly
            self.position_bias =  np.random.randint(0,self.n_positions)
        else:
            self.position_bias = self.rotation_steps # use preset rotation bias


        #the image at the beginning position is always kept
        self.kept_images.append(self.current_position)
    
        self.state_images[0] = self.current_position #add first image to state
        

        if self.dataset_path == '':
            #model = np.random.randint(20) #we use first 10 models from database for training
            model = random.choice(self.rnd_train_models) #take random  model from available models list
            self.spc = space_carving_rotation( os.path.join(MODELS_PATH,str(model).zfill(3)) , gt_mode=self.gt_mode, rotation_steps=self.position_bias,total_positions=self.n_positions)
        else:
            self.spc = space_carving_rotation(self.dataset_path, gt_mode=self.gt_mode,rotation_steps=self.position_bias,total_positions=self.n_positions)

        
        self.spc.carve(self.current_position)
        vol = self.spc.volume
        

        if self.gt_mode is True:
            self.last_gt_ratio = self.spc.gt_compare_solid()
        else:
            #get number of -1's (empty space), 0's (undetermined) and 1's (solid) from 3d volume
            self.h = [np.count_nonzero(vol == -1), np.count_nonzero(vol == 0), np.count_nonzero(vol == 1) ] 
            self.last_vspaces_count = self.h[0]   #spaces count from last sd volume carving


        #self.current_state = (vol.astype('float16') , self.state_images) #self.spc.sc.values().astype('float16')  

        #self.current_state = ( self.zeros_test , self.state_rel_images)
        #self.current_state = ( vol.astype('float16') , np.zeros(self.n_images))
        #self.current_state = ( self.zeros_test , np.zeros(self.n_images))
        #self.current_state = ( self.zeros_test , np.zeros(1))
        #self.current_state = ( np.random.randint(-1,1, size= self.volume_shape).astype('float16'), np.array([self.current_position],dtype=int))
        
        
        self.current_state = ( vol.astype('float16') , np.array([self.current_position],dtype=int))

        #self.current_state = vol.astype('float16')

        return self.current_state


    @property
    def nA(self):
        return self.action_space.n

    def render(self, mode='human', close=False):
        """
        :param mode:
        :return:
        """
        return

    def step(self, action):
        self.num_steps += 1
        
        #move n steps from current position
        steps = self.actions[action]
        self.current_position = self.calculate_position(self.current_position, steps)
        
        
        
        self.kept_images.append(self.current_position)
        #add image to position state
        self.state_images[self.num_steps] = self.current_position

        #carve in new position
        self.spc.carve(self.current_position)
        vol = self.spc.volume

        
        if self.gt_mode is True:
            #calculate increment of solid voxels ratios between gt and current volume
            gt_ratio = self.spc.gt_compare_solid()
            delta_gt_ratio = gt_ratio - self.last_gt_ratio
            self.last_gt_ratio = gt_ratio
            reward = delta_gt_ratio
        
        else:
            #get number of -1's (empty space), 0's (undetermined) and 1's (solid) from 3d volume
            self.h = [np.count_nonzero(vol == -1), np.count_nonzero(vol == 0), np.count_nonzero(vol == 1) ] #np.histogram(self.spc.sc.values(), bins=3)[0]
            #calculate increment of detected spaces since last carving
            delta = self.h[0] - self.last_vspaces_count
            self.last_vspaces_count = self.h[0]
            reward = min(delta,30000) / 30000
        

        if self.num_steps >= (self.n_images-1):
            #get number of -1's (empty space), 0's (undetermined) and 1's (solid) from 3d volume
            self.h = [np.count_nonzero(vol == -1), np.count_nonzero(vol == 0), np.count_nonzero(vol == 1) ]
            self.last_vspaces_count = self.h[0]
            self.done = True
           
        self.total_reward += reward

        
        #self.current_state = ( vol.astype('float16') , self.state_images)

        #self.current_state = ( self.zeros_test , self.state_rel_images)
        #self.current_state = ( vol.astype('float16') , np.zeros(self.n_images))
        #self.current_state = ( self.zeros_test , np.zeros(self.n_images))
        #self.current_state = ( self.zeros_test , np.zeros(1))
        
        self.current_state = ( vol.astype('float16') , np.array([self.current_position],dtype=int))

        #self.current_state = ( np.random.randint(-1,1, size= self.volume_shape).astype('float16'), np.array([self.current_position],dtype=int))
        #self.current_state = vol.astype('float16')

        return self.current_state, reward, self.done, {}

 
    def minMaxNorm(self,old, oldmin, oldmax , newmin , newmax):
        return ( (old-oldmin)*(newmax-newmin)/(oldmax-oldmin) ) + newmin

   
  
    def calculate_position(self,init_state,steps):
        n_pos = init_state + steps
        if n_pos>(self.n_positions-1):
            n_pos -= self.n_positions
        elif n_pos<0:
            n_pos += self.n_positions
        return n_pos


