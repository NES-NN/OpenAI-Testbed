import argparse
import gym
import ppaquette_gym_super_mario
import numpy as np
import shutil
import csv
import os


# -----------------------------------------------------------------------------
#  GLOBALS
# -----------------------------------------------------------------------------


game_name = 'ppaquette/SuperMarioBros-1-1-Tiles-v0'
smb_env = None
args = None


# -----------------------------------------------------------------------------
#  VINE LOGGING
# -----------------------------------------------------------------------------


def master_extract_cloud_ga(results, iteration):
    """Save offspring statistics"""
    path = args.loggingDir + "/snapshot_gen_{:04}/".format(int(iteration))
    if not os.path.exists(path):
        os.makedirs(path)

    filename = "snapshot_offspring_{:04}.dat".format(int(iteration))
    with open(os.path.join(path, filename), 'w+') as file:
        writer = csv.writer(file, delimiter=' ')
        for curr_task_results in results:
            #if (result == "distance"):
            # {'level': 0, 'distance': 6, 'score': 0, 'coins': 0, 'time': 388, 'player_status': 0, 'life': 3}
            
            #np is NumPy
            #looks like VINE wants floating point values.
            #2 values + fitness (in our case distance is fitness)
            row = np.hstack(("{:.6f}".format(curr_task_results.get('score')),"{:.8f}".format(curr_task_results.get('time')),"{:.6f}".format(curr_task_results.get('distance'))))
            writer.writerow(row)

    print('Created snapshot:' + filename)


def master_extract_parent(results, iteration):
    """Save parent statistics"""
    path = args.loggingDir + "/snapshot_gen_{:04}/".format(int(iteration +1))
    if not os.path.exists(path):
        os.makedirs(path)

    filename = "snapshot_parent_{:04}.dat".format(int(iteration +1))    
    with open(os.path.join(path, filename), 'w+') as file:
        writer = csv.writer(file, delimiter=' ')
        for curr_task_results in results:
            #if (result == "distance"):
            # {'level': 0, 'distance': 6, 'score': 0, 'coins': 0, 'time': 388, 'player_status': 0, 'life': 3}
            
            #np is NumPy
            #looks like VINE wants floating point values.
            row = np.hstack(("{:.6f}".format(curr_task_results.get('score')),"{:.8f}".format(curr_task_results.get('time')),"{:.6f}".format(curr_task_results.get('distance'))))            
            writer.writerow(row)

            #just need the first
            break

    print('Created parent snapshot:' + filename)


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

                if (strike > 3):
                    done = True
                
                last_distance = info.get('distance')

            child_run += 1
            print("Mock Child Run: {} of Gen: {} completed.".format(child_run, experiment))
            experiment_infos.append(info)

        if args.vineLogging:
            master_extract_cloud_ga(experiment_infos, experiment)
            master_extract_parent(experiment_infos, experiment)

        experiment += 1
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mario Random Trainer')
    parser.add_argument('--experiment-count', dest="experimentCount", type=int, default=10,
                        help="The amount of random loops to perform")
    parser.add_argument('--children-count', dest="childrenCount", type=int, default=20,
                        help="The amount of child loops to perform per top level loop")
    parser.add_argument('--vine-logging', dest="vineLogging", action='store_true',
                        help="Log out fitness of parent and children generations for VINE")
    parser.add_argument('--logging-dir', dest="loggingDir", type=str, default="snapshots",
                        help="The directory to log into")
    parser.add_argument('--display', dest="display", type=int, default=1,
                        help="The virtual display buffer to bind to.  Will only bind on positive integers")
    args = parser.parse_args()

    if args.display >= 1:
        os.environ["DISPLAY"] = ":{0}".format(args.display)

    smb_env = gym.make(game_name)
    random_moves(smb_env)
