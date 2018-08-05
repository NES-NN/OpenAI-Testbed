import gym

__all__ = ['SetSaveStateFolder']


def SetSaveStateFolder(stateFileLocation):
    """ set the folder for the saving and loading of game states"""

    class SetSaveStateFolderWrapper(gym.Wrapper):
        """
            State wrapper to set the path to the save game state 
        """
        def __init__(self, env):
            super(SetPlayingModeWrapper, self).__init__(env)
           
            if not os.path.isfile(stateFileLocation):
                logger.info("Could not load save file!")
                raise gym.error.Error('Error - Could not load save file! "{}" '.format(stateFileLocation))
            else:
                logger.info("state save file found.")
                
            self.unwrapped.stateFileLocation = stateFileLocation
            
    return SetSaveStateFolderWrapper
