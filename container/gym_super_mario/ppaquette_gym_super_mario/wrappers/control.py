import gym

__all__ = ['SetPlayingMode']


def SetPlayingMode(target_mode):
    """ target mode can be 'algo','human'  or 'normal' """

    class SetPlayingModeWrapper(gym.Wrapper):
        """
            Control wrapper to change playing mode 'human', 'algo' or 'normal'
        """
        def __init__(self, env):
            super(SetPlayingModeWrapper, self).__init__(env)
            if target_mode not in ['algo', 'human', 'normal']:
                raise gym.error.Error('Error - The mode "{}" is not supported. Supported options are "algo", "normal" or "human"'.format(target_mode))
            self.unwrapped.mode = target_mode

        def reset(self, **kwargs):
            self.env.reset(**kwargs)

    return SetPlayingModeWrapper
