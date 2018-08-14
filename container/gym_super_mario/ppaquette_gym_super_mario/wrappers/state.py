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
            self.baseSaveStateFile = "0-1.fcs"
            self.distance = 1
           
            if not os.path.isfile(saveStateFolder + self.baseSaveStateFile):
                raise gym.error.Error('Error - Could not find base state file.  Please check your save folder exists: "{}" and contains the expected save file: "{}" '.format(saveStateFolder, self.baseSaveStateFile))
                
            self.unwrapped.saveStateFolder = saveStateFolder

        def reset(self, **kwargs):
            # LoadState
            if self.unwrapped.shouldReloadFromSavedState:                
                if not os.path.isfile(self.unwrapped.saveStateFolder+ self.baseSaveStateFile):
                    raise gym.error.Error('Error - Save state folder now looks broken!?! "{}" '.format(saveStateFolder + self.baseSaveStateFile))
                
                self.unwrapped.loadState(self.unwrapped.saveStateFolder, self.distance)
                self.unwrapped.shouldReloadFromSavedState = False #done so reset flag

            return self.env.reset(**kwargs)

        def loadSaveStateFile(self, distance):
            self.unwrapped.shouldReloadFromSavedState = True
            self.distance = distance

        def saveToStateFile(self):
            self.unwrapped.saveState = True
            
    return EnableStateSavingAndLoadingWrapper
