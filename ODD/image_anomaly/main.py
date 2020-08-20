#!/usr/bin/python3
import os
import argparse
from scripts.engines.server_manager import ServerManagerBinary
from scripts.engines.setup_world import SetupWorld
from scripts.rl_agent.ddpg_agent import ddpgAgent
from scripts.rl_agent.input_preprocessor import InputPreprocessor
import numpy as np

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
        collect = args_assertions(args)
        carla_server = ServerManagerBinary({'CARLA_SERVER': os.environ["CARLA_SERVER"]})
        carla_server.reset()
        carla_server.wait_until_ready()
        env = SetupWorld(town=1, gui=args.gui, collect=collect, perception=args.perception)
        agent = ddpgAgent(Testing=args.testing)
        input_preprocessor = InputPreprocessor()
        for episode in range(args.episode):
            initial_distance = np.random.normal(100, 1)
            initial_speed = np.random.uniform(5,45)
            #initial_speed =45
            s = env.reset(initial_distance, initial_speed)
            print("Episode {} is started, target distance: {}, target speed: {}, initial distance: {}, initial speed: {}".format(episode, initial_distance, initial_speed, s[0], s[1]))
            s = input_preprocessor(s) 
            epsilon = 1.0 - (episode+1)/(args.episode)
            while True:
                a = agent.getAction(s, epsilon)
                s_, r, done= env.step(a[0][0])
                s_ = input_preprocessor(s_)
                if args.testing is False:
                    agent.storeTrajectory(s, a, r, s_, done)
                    agent.learn()
                s = s_

                if done:
                    print("Episode {} is done, the reward is {}".format(episode,r))
                    break
            #print("Out of break loop")
            if args.testing is False:
                if np.mod(episode, 10) == 0:
                    agent.save_model()
        carla_server.stop()
    
    except AssertionError as error:
        print(repr(error))
    except Exception as error:
        print('Caught this error: ' + repr(error))
        carla_server.stop()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        carla_server.stop()

