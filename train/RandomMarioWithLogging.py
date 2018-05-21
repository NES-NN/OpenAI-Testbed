import gym
import ppaquette_gym_super_mario
import numpy as np


def master_extract_cloud_ga(curr_task_results, iteration):
    import csv
    import os

    path = "snapshots/snapshot_gen_{:04}/".format(int(iteration))
    if not os.path.exists(path):
        os.makedirs(path)

    filename = "snapshot_offspring_{:04}.dat".format(int(iteration))
    with open(os.path.join(path, filename), 'w+') as file:
        writer = csv.writer(file, delimiter=' ')
        #for result in curr_task_results:
            #if (result == "distance"):
            # {'level': 0, 'distance': 6, 'score': 0, 'coins': 0, 'time': 388, 'player_status': 0, 'life': 3}
            
                #np is NumPy
        row = np.hstack((curr_task_results.get('distance'),curr_task_results.get('score')))
        writer.writerow(row)


env = gym.make('ppaquette/SuperMarioBros-1-1-v0')
observation = env.reset()
done = False
t = 0
while not done:
    action = env.action_space.sample()  # choose random action
    observation, reward, done, info = env.step(action)  # feedback from environment    
    t += 1
    if not t % 100:
        print(t, info)
        master_extract_cloud_ga(info,t)
        


