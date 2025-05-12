from collections import deque

import numpy as np
from godot_rl.wrappers.clean_rl_wrapper import CleanRLGodotEnv
from gymnasium.spaces import Box


class StateStack:
    """Wrapper of :class:`CleanRLGodotEnv`. Support stacking multiple states as observation.

    Return observation of shape (num_envs, num_stacks, *single_obs_shape)
    """

    def __init__(self, env: CleanRLGodotEnv, num_stacks: int, every_n: int = 1):
        """
        If `num_stacks` is 3 and `every_n` is 2, for stored states
        `[..., s-4, s-3, s-2, s-1, s-0]`, the final observation will be a
        stack of `[s-4, s-2, s-0]`.
        """
        self.env = env
        self.num_stacks = num_stacks
        self.every_n = every_n

        queue_len = (num_stacks - 1) * (every_n - 1) + num_stacks
        self.queue_len = queue_len
        self.states = deque(maxlen=queue_len)

    def reset(self, seed):
        obs, info = self.env.reset(seed=seed)
        for _ in range(self.queue_len):
            self.states.append(obs)
        return self._get_observation(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.states.append(obs)
        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self):
        selected_frames = []
        for i in range(self.num_stacks):
            idx = -1 - i * self.every_n
            selected_frames.append(self.states[idx])
        return np.stack(selected_frames, axis=1)

    @property
    def single_observation_space(self):
        space = self.env.single_observation_space
        if isinstance(space, Box):
            return Box(-1, 1, [self.num_stacks, *space.shape])
        else:
            raise NotImplementedError(f"Observation space {space} is not supported.")

    @property
    def single_action_space(self):
        return self.env.single_action_space

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    def close(self) -> None:
        self.env.close()
