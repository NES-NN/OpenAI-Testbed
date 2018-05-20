#!/usr/bin/python3

import gym
import gym_pull
gym_pull.pull('github.com/ppaquette/gym-super-mario')
env = gym.make('ppaquette/SuperMarioBros-1-1-v0')

for i_episode in range(100):
    observation = env.reset()
    while True:
        env.render()  
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)  
        print(observation, reward, done, info)
        if done:
            break
