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

        def reset(self, **kwargs):
            #LoadState
            if self.unwrapped.loadStateFromFile:                
                if not os.path.isfile(self.unwrapped.stateFileLocation):
                    raise gym.error.Error('NesEnv_Error - Could not load save file! "{}" '.format(self.unwrapped.stateFileLocation))
                
                self.unwrapped.loadState(self.unwrapped.stateFileLocation)
                self.unwrapped.loadStateFromFile = False #done so reset flag

            #Reload state -reload not supported since need to do a full load always!
            #if self.unwrapped.reloadState:                
            #    self.unwrapped.reloadLastSavedState()
            #    self.unwrapped.reloadState = False

            return self.env.reset(**kwargs)

        def loadSaveStateFile(self):
            self.unwrapped.loadStateFromFile = True

        def reloadSaveStateFile(self):
            self.unwrapped.reloadState = True

        def saveToStateFile(self):
            self.unwrapped.saveState = True
        
        # This is where we should intercept stuck and reset?
        #def _step(self, action):
        
        #   observation, reward, done, info = self.env.step(action)
        #   return observation, reward, done, info
            
    return SetSaveStateFolderWrapper
