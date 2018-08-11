import gym
import os

__all__ = ['SetSaveStateFolder']


def SetSaveStateFolder(saveStateFolder):
    """ set the folder for the saving and loading of game states"""

    class SetSaveStateFolderWrapper(gym.Wrapper):
        #this file is the start of the game (Level 1, no distance gained)
        baseSaveStateFile = "state-1-1.fcs"

        """
            State wrapper to set the path to the save game state 
        """
        def __init__(self, env):
            super(SetSaveStateFolderWrapper, self).__init__(env)
           
            if not os.path.isfile(saveStateFolder + baseSaveStateFile):
                raise gym.error.Error('Error - Could not load save file! "{}" '.format(saveStateFolder + baseSaveStateFile))
                
            self.unwrapped.saveStateFolder = saveStateFolder

        def reset(self, **kwargs):
            # LoadState
            if self.unwrapped.loadStateFromFile:                
                if not os.path.isfile(self.unwrapped.saveStateFolder+ baseSaveStateFile):
                    raise gym.error.Error('Error - Could not load save file! "{}" '.format(saveStateFolder + baseSaveStateFile))
                
                self.unwrapped.loadState(self.unwrapped.saveStateFolder+ baseSaveStateFile)
                self.unwrapped.loadStateFromFile = False #done so reset flag

            return self.env.reset(**kwargs)

        def loadSaveStateFile(self):
            self.unwrapped.loadStateFromFile = True

        def saveToStateFile(self):
            self.unwrapped.saveState = True
            
    return SetSaveStateFolderWrapper
