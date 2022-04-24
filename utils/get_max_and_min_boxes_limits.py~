import numpy as np
import json
#import tensorflow as tf
import os
import time
import matplotlib
import matplotlib.pyplot as plt


# Parent Directory path 
parent_dir = "/home/pico/uni/romi/rl_sony/arabidopsis_image_sets/"
paths = ['000_2d','001_2d','003_2d','006_2d','009_2d','124_2d','195_2d',]


minmaxx = [1000,-1000]
minmaxy = [1000,-1000]
minmaxz = [1000,-1000]

for p in paths:
   boxf = os.path.join(parent_dir, p , 'bbox.json')
   box = json.load(open(boxf))
   print(box['x'],box['y'],box['z'])
   minmaxx[0] = min(minmaxx[0],box['x'][0]) 
   minmaxx[1] = max(minmaxx[1],box['x'][1]) 
   minmaxy[0] = min(minmaxy[0],box['y'][0]) 
   minmaxy[1] = max(minmaxy[1],box['y'][1]) 
   minmaxz[0] = min(minmaxz[0],box['z'][0]) 
   minmaxz[1] = max(minmaxz[1],box['z'][1]) 
print('**************')
print(minmaxx,minmaxy,minmaxz)
