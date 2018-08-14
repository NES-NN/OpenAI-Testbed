import argparse
import gym
import ppaquette_gym_super_mario
import numpy as np
import shutil
import csv
import os
import logging
import sys

from ppaquette_gym_super_mario.wrappers import *

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


# -----------------------------------------------------------------------------
#  GLOBALS
# -----------------------------------------------------------------------------


game_name = 'ppaquette/SavingSuperMarioBros-1-1-Tiles-v0'
smb_env = None
args = None

# -----------------------------------------------------------------------------
#  RANDOM
# -----------------------------------------------------------------------------


def random_moves(env):
    """Execute random moves on the network"""
    experiment = 0

    while (experiment < args.experimentCount):
        child_run = 0
        experiment_infos = []

        while (child_run < args.childrenCount):    

            done = False
            t = 0
            last_distance = 0;
            strike = 0;
            best_distance = 40;

            #env.reloadSaveStateFile()  --not supported in gym env.
            env.loadSaveStateFile(best_distance) #passes the saveState filename to lua, the reset command will
            #trigger the loading of the state file.

            observation = env.reset()
            

            while not done:
                # Choose random action
                action =  np.random.randint(2,size=env.action_space.shape[0])
                
                # Feedback from environment  
                observation, reward, done, info = env.step(action)  
                t += 1
                
                if not t % 100:
                    logger.info("{} {}".format(t, info))

                # Add cutoff to none progressing random agents
                if (info.get('distance') < last_distance):
                    strike += 1
                else:        
                    strike = 0
                    
                if (strike > 6):                     
                    done = True
                
                last_distance = info.get('distance') 
                if ((last_distance - best_distance) > 50):  #save every 50 step gain
                    best_distance = last_distance
                    logger.info("New Best distance {}... saving again".format(best_distance))
                    env.saveToStateFile()

            child_run += 1
            logger.info("Mock Child Run: {} of Gen: {} completed.".format(child_run, experiment))
            experiment_infos.append(info)
            
        experiment += 1
        
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mario Random Trainer')
    parser.add_argument('--experiment-count', dest="experimentCount", type=int, default=10,
                        help="The amount of random loops to perform")
    parser.add_argument('--children-count', dest="childrenCount", type=int, default=20,
                        help="The amount of child loops to perform per top level loop")
    parser.add_argument('--snapshots-dir', dest="snapshotsDir", type=str, default="/opt/train/Random/snapshots",
                        help="The snapshots directory for VINE logging")
    args = parser.parse_args()

    os.environ["DISPLAY"] = ":1"
    
    gym.undo_logger_setup()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info('Starting saving state training example')

    smb_env = gym.make(game_name)
    
    #We need to have some distance+generation.fcs file creator step
    wrapper = EnableStateSavingAndLoading('/opt/train/stateSaving/saveStates/')

    #game was going too fast for me via remote connection (3200%!)
    wrapper2 = SetPlayingMode('normal')

    smb_env = wrapper(wrapper2(smb_env))

    random_moves(smb_env)
