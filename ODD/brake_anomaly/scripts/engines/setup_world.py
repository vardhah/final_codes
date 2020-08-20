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

        self.mu_l=0.0         # angular acceleration
        self.mu=0.0
        self.bs=0.8         #set braking value at which it switch fron static to kinetic
        self.distance=0.0  
        self.velocity=0.0 

        self.episode = 0
        self.flag=0
        self.kickspd_e1=0.0
        self.kickspd_e2=0.0
        self.crossing_velocity=0.0
        self.friction_of_patch=0.0
        self.variance_friction=0.0
        self.collect = collect
        self.distance_rear_car=0.0
        self.velocity_rear_car=0.0
        self.currloc_rearcar=0.0
        self.flagdone1=0
        self.flagdone2=0
        self.action1=0
        self.geolocation=0
        self.shift=0.0
        self.step_count=0
    def reset(self, leadcar_loc,initial_speed,friction):  
        self.kickspd_e1=initial_speed
        self.friction_of_patch=friction
        self.mu=0.0
        self.rewards = reward_calc(a=1.0,d=1.0,base=1.9)
        self.distance=self.geolocation=leadcar_loc                           #distance means => obstacle distance


        if self.collect["option"] != 0 and self.flag!=1 :
            self.flag=1
            self.episode=0 
            self.collect_data = collectData(self.collect["path"])
            
        
        self.step_count = 0
        self.velocity= initial_speed *4/9     # convert velocity is m/s  
        vehicle_stop=self.velocity<=0.0
        self.episode+=1
        self.flagdone1=1
        return [self.distance, self.velocity ,self.mu]
    
    def step(self, action):
        step_of_obstacle=0
        self.step_count+=1
        self.action1 = action       
        self.shift+=step_of_obstacle
        if action<=0.8: 
            self.mu_l=(2* self.friction_of_patch*self.bs *action)/(np.square(self.bs)+ np.square(action))
        elif action>0.8:
            self.mu_l = 0.7* self.friction_of_patch
        

        accel= -1*self.mu_l*self.g
        #print('mu_s is',mu_s)
        distance_travelled= self.velocity *self.dt +0.5*accel*np.square(self.dt) 
        self.distance= self.distance-distance_travelled-step_of_obstacle
        self.geolocation= self.geolocation-distance_travelled
        self.velocity= self.velocity+accel*self.dt
        friction_force= self.mu_l*self.mass*self.g          #braking force
        
        
        if self.distance >5 and self.distance<10:
            self.crossing_velocity= self.velocity         
        #print('current velocity is :',self.velocity)
    
        isStop = self.velocity <= 0.0005
        isCollision = self.distance<=0
        done = isStop or isCollision
        
        if done and self.flagdone1==1: 
            self.flagdone1=0
            groundtruth_distance=self.geolocation
            reward = self.rewards.reward_total(groundtruth_distance,self.crossing_velocity)
            print("Total shift :",self.shift)
            print(" Ego stopped & Episode: {},,KickSpd:{}, CRS_spd: {},Reward: {}, Stop_Dist: {}".format(self.episode,self.kickspd_e1,self.crossing_velocity,reward, groundtruth_distance))     
            print("Time of simulation:",self.step_count/20 ,"Seconds & stepcount is",self.step_count)
        else:
            reward = 0     
        return [[self.distance, self.velocity,self.mu_l], reward, done] 
    
    def collectdata(self):
        if self.collect["option"] != 0:
               #self.collect_data(self.episode, self.kickspd_e1,self.distance,self.velocity,self.action1, self.kickspd_e2,self.currloc_rearcar, self.distance_rear_car,self.velocity_rear_car,self.action2)
               self.collect_data(self.geolocation,self.velocity,self.mu_l,self.action1,self.shift)
    
    def closefile(self):
        self.collect_data.close_csv()
