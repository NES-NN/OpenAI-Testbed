import gym
import ppaquette_gym_super_mario
import numpy as np
import shutil

#save offspring stats
#ugly name to match VINE examples for now
def master_extract_cloud_ga(allChildrenResults, iteration):
    import csv
    import os

    path = "snapshots/snapshot_gen_{:04}/".format(int(iteration))
    if not os.path.exists(path):
        os.makedirs(path)

    filename = "snapshot_offspring_{:04}.dat".format(int(iteration))
    with open(os.path.join(path, filename), 'w+') as file:
        writer = csv.writer(file, delimiter=' ')
        for curr_task_results in allChildrenResults:
            #if (result == "distance"):
            # {'level': 0, 'distance': 6, 'score': 0, 'coins': 0, 'time': 388, 'player_status': 0, 'life': 3}
            
            #np is NumPy
            #looks like VINE wants floating point values.
            #2 values + fitness (in our case distance is fitness)
            row = np.hstack(("{:.6f}".format(curr_task_results.get('score')),"{:.8f}".format(curr_task_results.get('time')),"{:.6f}".format(curr_task_results.get('distance'))))
            writer.writerow(row)
    print('Created snapshot:' + filename)

#save parent stats
#same ugly name issue for now
def master_extract_parent(allChildrenResults, iteration):
    import os
    import csv
    
    #Since all runs are equivalent
    #We will take the first result and pretend it is the parent
    path = "snapshots/snapshot_gen_{:04}/".format(int(iteration +1))
    if not os.path.exists(path):
        os.makedirs(path)

    filename = "snapshot_parent_{:04}.dat".format(int(iteration +1))    
    with open(os.path.join(path, filename), 'w+') as file:
        writer = csv.writer(file, delimiter=' ')
        for curr_task_results in allChildrenResults:
            #if (result == "distance"):
            # {'level': 0, 'distance': 6, 'score': 0, 'coins': 0, 'time': 388, 'player_status': 0, 'life': 3}
            
            #np is NumPy
            #looks like VINE wants floating point values.
            row = np.hstack(("{:.6f}".format(curr_task_results.get('score')),"{:.8f}".format(curr_task_results.get('time')),"{:.6f}".format(curr_task_results.get('distance'))))            
            writer.writerow(row)
            break #just need the first
    print('Created parent snapshot:' + filename)
    

env = gym.make('ppaquette/SuperMarioBros-1-1-v0')
experiment = 0

#loop experiments forever
while True: 

    #pretend we have 20 children
    childRun = 0
    experimentInfos = [] #store all the child results
    while (childRun < 20):    
        observation = env.reset()
        done = False
        t = 0
        lastDistance = 0;
        strike = 0;
        while not done:
            action =  np.random.randint(2,size=env.action_space.shape[0])  # choose random action
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
        childRun += 1
        print("Mock Child Run: {} of Gen: {} completed.".format(childRun ,experiment))
        experimentInfos.append(info)

    master_extract_cloud_ga(experimentInfos,experiment)
    master_extract_parent(experimentInfos,experiment)

    experiment += 1
            
    

