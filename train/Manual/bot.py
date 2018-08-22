import gym
from ppaquette_gym_super_mario.wrappers import *


def get_action (observation):
    flat = observation.flatten()

    try:
        # Get Mario's location
        mario_location = flat.tolist().index(3)

        # Check what is 1 & 2 blocks in front of mario.
        obj = [flat[mario_location + 2], flat[mario_location + 3]]

        # If it is a pipe or enemy Jump!
        if any(i in obj for i in [1, 2]):
            print("Jump!")
            return [0, 0, 0, 1, 1, 0]
        # if it is empty space, move forward.
        else:
            return [0, 0, 0, 1, 0, 0]

    # Do nothing if Mario isn't on the screen yet.
    except ValueError:
        return [0, 0, 0, 0, 0, 0]


def play(e):
    action = [0, 0, 0, 0, 0, 0]
    done = False

    e.reset()

    while not done:
        observation, reward, done, info = e.step(action)

        action = get_action (observation)


def main():
    # os.environ["DISPLAY"] = ":1"

    wrapper = SetPlayingMode('normal')
    e = wrapper(gym.make('ppaquette/SuperMarioBros-1-1-Tiles-v0'))

    play(e)


if __name__ == '__main__':
    main()
