import gym
import os

__all__ = ['SetSaveStateFolder']


def SetSaveStateFolder(stateFileLocation):
    """ set the folder for the saving and loading of game states"""

    class SetSaveStateFolderWrapper(gym.Wrapper):
        """
            State wrapper to set the path to the save game state 
        """
        def __init__(self, env):
            super(SetSaveStateFolderWrapper, self).__init__(env)
           
            if not os.path.isfile(stateFileLocation):
                raise gym.error.Error('Error - Could not load save file! "{}" '.format(stateFileLocation))
                
            self.unwrapped.stateFileLocation = stateFileLocation

        def reset(self):
            return self.env.reset()

        def loadSaveStateFile(self):
            self.unwrapped.loadStateFromFile = True

        def reloadSaveStateFile(self):
            self.unwrapped.reloadState = True

        def saveToStateFile(self):
            self.unwrapped.saveState = True
        
        def _step(self, action):
        # This is where we should intercept stuck and reset?
            observation, reward, done, info = self.env.step(action)
            return observation, reward, done, info
            
    return SetSaveStateFolderWrapper
