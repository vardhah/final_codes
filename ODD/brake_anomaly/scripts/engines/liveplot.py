#!/usr/bin/python3
import random
from itertools import count
import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
#plt.style.use('fivethirtyeight')

class Liveplot():
    def __init__(self):
       self.fig = plt.figure()
       self.ax1 = plt.axes(xlim=(0, 130), ylim=(-10, 120))
       self.xs=[]
       self.ys=[]
       self.obstacle=[]
       self.cnt=0
    
    def __call__(self):
       t_data = pd.read_csv('./DATA/Data.csv')
       self.data=t_data[["dist_o1","shift_obstacle"]]       #
       print("Numbers of frames:",self.data.shape[0])
       print("Data is:",self.data)
       iterator=np.arange(self.data.shape[0]-1)
       print("iterator is:",iterator )
       ani = FuncAnimation(self.fig, self.animate, frames=np.arange(2),interval=200,repeat=False)
       
       plt.show()
       ani.save('myAnimation.mp4',writer='ffmpeg')
       #ani.save('sim_result.mp4', writer='imagemagick')

    def animate(self,i):
     #self.obstacle.append(self.data['shift_obstacle'].values[self.cnt])
     flag=0
     print("i is:",i)
     self.ys.append(self.data['dist_o1'].values[self.cnt])
     self.xs.append(self.cnt)
     self.ax1.clear()
     
     print("--> New data is:",self.data['dist_o1'].values[self.cnt],"count is:",self.cnt)
     """
     self.ax1.annotate('Starting', xy =(50, 1), 
             xytext =(100, 1.8), 
             arrowprops = dict(facecolor ='green', 
                               shrink = 0.05),   )  
     """
     plt.xlim(0, self.data.values.shape[0]); plt.ylim(-10, 120)
     plt.plot(self.xs, self.ys, label='car_position')
     plt.hlines(self.data['shift_obstacle'].values[self.cnt],1,self.data.values.shape[0],'r',label='obstacle_position')
     plt.legend(loc='upper right')
     plt.grid()
     self.cnt+=1

     #plt.tight_layout() 
     #self.cnt+=1

