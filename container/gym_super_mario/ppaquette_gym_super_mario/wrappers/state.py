import gym
import os

__all__ = ['EnableStateSavingAndLoading']


def EnableStateSavingAndLoading(saveStateFolder):
    """ set the folder for the saving and loading of game states"""

    class EnableStateSavingAndLoadingWrapper(gym.Wrapper):

        """
            State wrapper to set the path to the save game state 
        """
        def __init__(self, env):
            super(EnableStateSavingAndLoadingWrapper, self).__init__(env)

            #this file is the start of the game (Level 1, no distance gained)
            self.baseSaveStateFile = "state-1-1.fcs"
           
            if not os.path.isfile(saveStateFolder + self.baseSaveStateFile):
                raise gym.error.Error('Error - Could not load save file! "{}" '.format(saveStateFolder + self.baseSaveStateFile))
                
            self.unwrapped.saveStateFolder = saveStateFolder

        def reset(self, **kwargs):
            # LoadState
            if self.unwrapped.loadStateFromFile:                
                if not os.path.isfile(self.unwrapped.saveStateFolder+ self.baseSaveStateFile):
                    raise gym.error.Error('Error - Could not load save file! "{}" '.format(saveStateFolder + self.baseSaveStateFile))
                
                self.unwrapped.loadState(self.unwrapped.saveStateFolder+ self.baseSaveStateFile)
                self.unwrapped.loadStateFromFile = False #done so reset flag

            return self.env.reset(**kwargs)

        def loadSaveStateFile(self):
            self.unwrapped.loadStateFromFile = True

        def saveToStateFile(self):
            self.unwrapped.saveState = True
            
    return EnableStateSavingAndLoadingWrapper
