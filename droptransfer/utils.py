import cv2
import gym
import numpy as np
from gym import Wrapper


def process_state(state, trailing_dimension=False):
    """
    Convert to gray and shrink original state.

    Args:
        state: A [210, 160, 3] Atari RGB State of unsigned ints.
        trailing_dimension: If true, add a final channel dimension of size 1.

    Returns:
        A processed [84, 84[, 1]] state representing grayscale values.
    """

    state_gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)

    state_resized = cv2.resize(state_gray, (84, 110), interpolation=cv2.INTER_CUBIC)

    state_cropped = state_resized[18:102]

    if trailing_dimension:
        state_cropped = state_cropped[:, :, None]

    return state_cropped


class AtariEnvWrapper(Wrapper):
    """
  Wraps an Atari environment to do each action `n` number of times and to end an episode when a life is lost.
  """

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, *args, **kwargs):

        # Open AI defaults to skipping between 2 and 4 frames

        lives_before = self.env.ale.lives()
        next_state, reward, done, info = self.env.step(*args, **kwargs)
        lives_after = self.env.ale.lives()

        # End the episode when a life is lost
        if lives_before > lives_after:
            done = True

        # Clip rewards to [-1,1]
        reward = max(min(reward, 1), -1)

        return next_state, reward, done, info

    def reset(self, **kwargs):

        return self.env.reset(**kwargs)


def atari_make_initial_state(state):
    return np.stack([state] * 4, axis=2)


def atari_make_next_state(state, next_state):
    # Keeping track of last four frames. Each time head frame is bumped off.
    return np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)


def make_env(env_name, wrap=True):
    env = gym.make(env_name)
    # remove the timelimitwrapper
    env = env.env
    if wrap:
        env = AtariEnvWrapper(env)
    return env