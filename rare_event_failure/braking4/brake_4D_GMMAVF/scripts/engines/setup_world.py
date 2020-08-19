import sys
import logging
import os
import queue
import random
from scripts.engines.reward_calc import reward_calc
from scripts.engines.collect_data import collectData
import math
import numpy as np
import cv2

class SetupWorld():
    def __init__(self, mass=1300, wheel_radius=0.04, dt=0.05, collect={"option":0, "path": None}):
        self.mass=1300 #Kg
        self.r=0.04    # 16 inch=40 cm 
        self.dt=0.05   # frequency = 20 Hz 
        self.g= 9.8    # graviataional acceleration (in m/s^2)
        self.j=1       # kg-m^2 (Moment of Intertia of tire)

        self.mu=0.0         # Effective frictional coefficient
        self.bs=0.8         # Set braking value at which it switch fron static to kinetic
        self.distance=0.0  
        self.velocity=0.0 

        self.episode = 0
        self.flag=0
        self.kickspd=0.0
        self.crossing_velocity=0.0
        self.default_friction=0.0
        self.friction_of_patch=0.0
        self.location_of_patch=0.0
        self.size_of_patch=0.0 
        self.collect = collect
        self.reward_total=0.0
        self.rewards = reward_calc(a=1.0,d=1.0,base=1.9)
        self.counter=0
    def reset(self, initial_distance, initial_speed,friction,friction_patch,size_patch, location_patch):  
        self.kickspd=initial_speed
        self.default_friction=friction
        self.friction_of_patch=friction_patch
        self.size_of_patch=size_patch
        self.location_of_patch=location_patch
        self.mu=0.0
        self.distance=initial_distance
        self.reward_total=0.0
        #print('in reset')
        if self.collect["option"] != 0 and self.flag!=1 :
            #print ('in flag loop')
            self.flag=1
            self.episode=0 
            self.collect_data = collectData(self.collect["path"])
            
        
        self.step_count = 0
        self.velocity= initial_speed *4/9     # convert velocity is m/s   
        vehicle_stop=self.velocity<=0.0
        

        return [self.distance, self.velocity ,self.mu]
    
    def step(self, action):
        self.step_count += 1
       
        if self.distance>=(self.location_of_patch-self.size_of_patch) and self.distance<=(self.location_of_patch+self.size_of_patch) :
              friction=self.friction_of_patch
        else:
              friction= self.default_friction

        if action<=0.8: 
            self.mu=(2* friction*self.bs *action)/(np.square(self.bs)+ np.square(action))
        elif action>0.8:
            self.mu = 0.7* friction
        


# rest of dynamics, calculation 
        accel= -1*self.mu*self.g
        #print('mu_s is',mu_s)
        distance_travelled= self.velocity *self.dt +0.5*accel*np.square(self.dt) 
        self.distance= self.distance-distance_travelled
        self.velocity= self.velocity+accel*self.dt
        friction_force= self.mu*self.mass*self.g          #braking force
        
        isStop = self.velocity <= 0.05
        isCollision = self.distance<=0
        done = isStop or isCollision
        #print('Done is:',done)
        if done==False:
         #print('I am in done') 
         if self.distance>5: 
          reward = -1*accel*self.dt*2
         else: 
          reward= -1*(2**(5-self.distance))
        
        if self.distance >5 and self.distance<10:
            self.crossing_velocity= self.velocity         
        #print('current velocity is :',self.velocity)
    
       
        
        
        
        if done: 
            #print('In done')
            groundtruth_distance=self.distance
            reward = self.rewards.reward_t(groundtruth_distance,self.crossing_velocity)
        
        self.reward_total= self.reward_total+reward; 

        if done:
            self.counter+=1
            #if self.distance<=0:
            print("====> Episode: {},Spd:{}, friction: {},size:{},loc:{},Reward: {}, Stop_Dist: {}".format(self.episode,self.kickspd,self.friction_of_patch,self.size_of_patch,self.location_of_patch,self.reward_total, groundtruth_distance))
            if self.collect["option"] != 0:
               self.collect_data(self.episode, self.kickspd, self.friction_of_patch,self.default_friction,self.location_of_patch,self.size_of_patch, self.reward_total,groundtruth_distance)
            self.episode+=1
        
          

        return [[self.distance, self.velocity,friction], reward, done] 


    def closefile(self):
        if self.collect["option"] != 0:
          self.collect_data.close_csv()
