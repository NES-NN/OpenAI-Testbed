import gym
import ppaquette_gym_super_mario

env = gym.make('ppaquette/SuperMarioBros-1-1-Tiles-v0')
observation = env.reset()
done = False
t = 0


while not done:
    action = env.action_space.sample()  # choose random action
    observation, reward, done, info = env.step(action)  # feedback from environment
    t += 1
    if not t % 100:
        print(t, info)

