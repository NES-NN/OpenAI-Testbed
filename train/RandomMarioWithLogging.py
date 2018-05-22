import gym
import ppaquette_gym_super_mario
import numpy as np
import shutil

#save offspring stats
#ugly name to match VINE examples for now
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
    print('Created snapshot:' + filename)

#save parent stats
#same ugly name issue for now
def master_extract_parent(iteration):
    import os
    path = "snapshots/snapshot_gen_{:04}/".format(int(iteration))
    if not os.path.exists(path):
        os.makedirs(path)

    filename = "snapshot_parent_{:04}.dat".format(int(iteration))    
  
    previousPath = "snapshots/snapshot_gen_{:04}/".format(int(iteration - 1))
    previousFilename = "snapshot_offspring_{:04}.dat".format(int(iteration -1))
    
    shutil.copy2(previousPath + previousFilename, path + filename) 
    print('Created snapshot:' + filename)

env = gym.make('ppaquette/SuperMarioBros-1-1-v0')
experiment = 0;

#loop experiments forever
while True:    
    observation = env.reset()
    done = False
    t = 0
    lastDistance = 0;
    strike = 0;
    while not done:
        action = env.action_space.sample()  # choose random action
        observation, reward, done, info = env.step(action)  # feedback from environment    
        t += 1
        
        if not t % 100:
            print(t, info)

        #add cutoff to none progressing random agents
        if (info.get('distance') < lastDistance):
            strike +=1
        else:        
            strike = 0

        if (strike > 3):
            done = True
        
        lastDistance = info.get('distance')

    master_extract_cloud_ga(info,experiment)
    if (experiment > 0):
        master_extract_parent(experiment)

    experiment += 1
            
    

