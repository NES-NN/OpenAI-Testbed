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
            
    return SetSaveStateFolderWrapper
