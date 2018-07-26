import argparse
import gym
import ppaquette_gym_super_mario
import numpy as np
import shutil
import csv
import os
import logging


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
            observation = env.reset()

            done = False
            t = 0
            last_distance = 0;
            strike = 0;

            while not done:
                # Choose random action
                action =  np.random.randint(2,size=env.action_space.shape[0])
                
                # Feedback from environment  
                observation, reward, done, info = env.step(action)  
                t += 1
                
                if not t % 100:
                    print(t, info)

                # Add cutoff to none progressing random agents
                if (info.get('distance') < last_distance):
                    strike += 1
                else:        
                    strike = 0

                if (strike > 30): #now that the gym catches the stuck state, this doesn't make sense. 
                    #however leaving it at 3 would mean killing it before the gym state reload kicks in.
                    done = True
                
                last_distance = info.get('distance')

            child_run += 1
            print("Mock Child Run: {} of Gen: {} completed.".format(child_run, experiment))
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

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info('Starting saving state training example')

    smb_env = gym.make(game_name)
    
    random_moves(smb_env)
    
