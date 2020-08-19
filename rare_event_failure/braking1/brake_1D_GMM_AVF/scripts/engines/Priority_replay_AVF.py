import numpy as np 
import pandas as pd

class PR_AVF():
    def __init__(self,episode):
        self.i=-1
        self.episode=episode
        self.crashed = pd.read_csv('./DATA/crashed.csv')
        self.crashed=self.crashed.loc[self.crashed['Episode'] <= episode]
        self.crashed=self.crashed['Kick_Speed'].values
        print('Total crashed cases in this duration :',self.crashed.shape[0])
        
    def pr_sampler(self):
        self.i=self.i+1
        return  self.crashed[-1-self.i]


