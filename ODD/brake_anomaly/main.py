#!/usr/bin/python3
import os
import argparse
from scripts.engines.setup_world import SetupWorld
from scripts.rl_agent.ddpg_agent import ddpgAgent
from scripts.rl_agent.input_preprocessor import InputPreprocessor
import numpy as np
from scripts.engines.rrcf import rcf
from scripts.engines.liveplot import Liveplot
import matplotlib.pyplot as plt

def args_assertions(args):
    collect_1 = args.collect_perception is not None
    collect_2 = args.collect_detector is not None
    collect = {"option": 0, "path": None}
    assert (collect_1 and collect_2) != True, "don't set collect_detector and collect_perception simultaneously"
    if collect_1:
        collect = {"option": 1, "path": args.collect_perception} # collect data for perception training
    elif collect_2:
        collect = {"option": 2, "path": args.collect_detector} # collect data for detector training
    return collect

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Assurance Monitoring for RL-based emergency braking system.')
    parser.add_argument("-g", "--gui", help="set gui mode.", action="store_true")
    parser.add_argument("-t", "--testing", help="set testing mode", action="store_true", default=False)
    parser.add_argument("-cp", "--collect_perception", help="collect the data for perception training")
    parser.add_argument("-ca", "--collect_detector", help="collect the data for detector training")
    parser.add_argument("-p", "--perception", help="set the path of perception neural network")
    parser.add_argument("-e", "--episode", help="set the number of episode", type=int, default=1)

    args = parser.parse_args()

    try:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used
        collect = args_assertions(args)
        env = SetupWorld(mass=1300, wheel_radius=0.04, dt=0.05, collect=collect)
        agent = ddpgAgent(Testing=args.testing)
        input_preprocessor = InputPreprocessor()
        rcf=rcf(100)  
        print('Number of episodes :',args.episode)
        plot=Liveplot()
        cnt=0
        coDisp=[]
        #rcf.trainer()
        for episode in range(args.episode):
          #stopdist=1.0;	
          #Comment below 3 lines while training , its only for running set of testing 
          #print('***********************************************************************************************************')
          #print('******** Launching failure search test ID:',episode+1)
          #np.random.seed()
          #while stopdist>0:
            ##select parameters---------------------------------------------------------------    
            emergency_brake=False; anomaly_detector=True
            location_leadcar = 100; 
            velocity_lead = np.random.uniform(40,70)                            # Vanilla sampling                           
            #if velocity_lead <20 : velocity_lead=20
            #f velocity_lead >60 : velocity_lead=75 
            velocity_lead=70

            #print('spawned velocity is',velocity_lead)
            friction=0.9
            R = env.reset(location_leadcar,velocity_lead,friction)
            #print('came from SetupWorld')
            s1=R[0:3]
            #print('s1 is :',s1)
            #print('s2 is :',s2)
            s1 = input_preprocessor(s1)
            #print('Out from input preprocessor')
            epsilon = 1.0 - (episode+1)/(args.episode)
            anomaly=False
            threshold=0
            while True:
            # action prediction for rear vehicle(1) 
                 cnt+=1
                 #print(anomaly)
                 if emergency_brake==True : 
                  if (anomaly==False):
                    a = agent.getAction(s1, epsilon) 
                  else:
                    print("-->Anomalous data detected, Invoking emergency braking")
                    a=[[0.79]]  
                 else:
                     a = agent.getAction(s1, epsilon)  
            #detect anomaly 
            #kick in emergemcy braking     
                 s1_, r1, done1= env.step(a[0][0])
                 #print('State1 is',s1_,'action is:',a,"Done is",done1)
                 s1_ = input_preprocessor(s1_)
                 if args.testing is False:
                    agent.storeTrajectory(s1, a, r, s1_, done)
                    agent.learn()
                    #print('In learing loop')
                 s1 = s1_
                 env.collectdata()
                 if anomaly_detector==True:
                   s_original=input_preprocessor.unscaled_data(s1)
                   predictor_out=rcf.predictor(s_original)
                   coDisp.append(predictor_out[0])
                   print(predictor_out[0], "---",predictor_out[1])
                   if (predictor_out[0]-predictor_out[1]>threshold):
                     
                     anomaly=True
                   else:
                     anomaly=False
                   rcf.delete_node()
                 #if cnt>5:
                   #plot()
            #collecting data
                #env.collectdata()
            #Terminating Episode    
                 if done1:
                    stopdist=s1[0]*120
                    break

            if(stopdist<=0):
                print(' => => -------------   YEAH !!!  Found a crashed case')
                #env.collectdata()
            if args.testing is False:
                if np.mod(episode, 100) == 0:
                    agent.save_model()
                if np.mod(episode,500) == 0:
                    agent.save_intermittent_model(episode)
        env.closefile()
        print('finished')
        print("Plotting simulation results")
        #plot()
        np.savetxt("foo.csv", coDisp, delimiter=",")
        plt.plot(coDisp)
        
        
    except AssertionError as error:
        print(repr(error))
    except Exception as error:
        print('Caught this error: ' + repr(error))
    except KeyboardInterrupt:
        print("KeyboardInterrupt")


