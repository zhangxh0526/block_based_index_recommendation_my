import gym
import numpy as np
from gym import spaces


class ColumnGroupEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, block_num, index_num, query_num, workload_dim, block_dim, space_size, occupied_size ):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.MultiDiscrete([block_num, index_num])
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Dict({
            'workload':spaces.MultiDiscrete([query_num, workload_dim]),
            'block':spaces.MultiDiscrete([query_num, block_num, block_dim]),
            'candidate':spaces.MultiDiscrete([block_num, index_num]),
            'space':spaces.MultiDiscrete(space_size, occupied_size),
        })

    def step(self, action):
        ...
        return observation, reward, done, info

    def reset(self):
        ...
        return observation  # reward, done, info can't be included

    def render(self, mode="human"):
        ...

    def close(self):
        ...
