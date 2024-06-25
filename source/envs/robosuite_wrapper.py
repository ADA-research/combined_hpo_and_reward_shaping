# This file includes code from Gymnasium (https://github.com/Farama-Foundation/Gymnasium)
# Licensed under the MIT License (https://github.com/Farama-Foundation/Gymnasium/blob/main/LICENSE).

from typing import Dict
from envs.gym_wrapper import GymWrapper
import numpy as np

from gymnasium import spaces
from gymnasium.core import Env
from gymnasium.wrappers.time_limit import TimeLimit
from gymnasium.envs.registration import EnvSpec

import robosuite_wrapper as suite
from robosuite_wrapper.controllers import load_controller_config
from robosuite_wrapper.wrappers import Wrapper


class RobosuiteGymWrapper(Wrapper, Env):
    def __init__(self, env, keys=None):
        super().__init__(env=env)
        robots = "".join([type(robot.robot_model).__name__ for robot in self.env.robots])
        self.name = robots + "_" + type(self.env).__name__

        self.reward_range = (0, self.env.reward_scale)

        if keys is None:
            keys = []
            if self.env.use_object_obs:
                keys += ["object-state"]
            for idx in range(len(self.env.robots)):
                keys += ["robot{}_proprio-state".format(idx)]
        self.keys = keys

        self.env.spec = None
        self.metadata = None
        self._last_img = None

        obs = self.env.reset()
        if self.env.use_camera_obs:
            self._last_img = obs.pop("frontview_image")
        self.modality_dims = {key: obs[key].shape for key in self.keys}
        flat_ob = self._flatten_obs(obs)
        self.obs_dim = flat_ob.size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low=low, high=high)
        low, high = self.env.action_spec
        self.action_space = spaces.Box(low=low, high=high)

    def _flatten_obs(self, obs_dict, verbose=False):
        ob_lst = []
        for key in self.keys:
            if key in obs_dict:
                if verbose:
                    print("adding key: {}".format(key))
                ob_lst.append(np.array(obs_dict[key]).flatten())
        return np.concatenate(ob_lst)

    def reset(self):
        ob_dict = self.env.reset()
        if self.env.use_camera_obs:
            self._last_img = ob_dict.pop("frontview_image")
        return self._flatten_obs(ob_dict), {}

    def step(self, action):
        ob_dict, reward, done, info = self.env.step(action)
        if self.env.use_camera_obs:
            self._last_img = ob_dict.pop("frontview_image")
        return self._flatten_obs(ob_dict), reward, done, False, info

    def seed(self, seed=None):
        if seed is not None:
            try:
                np.random.seed(seed)
            except:
                TypeError("Seed must be an integer type!")

    def compute_reward(self):
        return self.env.reward()

    def render(self):
        if self.env.has_renderer:
            self.env.render()
        elif self.env.use_camera_obs:
            return np.flipud(self._last_img)


class TimeLimitLift(TimeLimit):
    def __init__(self, config: Dict):
        env = suite.make(
            env_name="Lift",
            robots="Panda",
            has_renderer=False,  #True if config['render_mode'] == 'human' else False,
            has_offscreen_renderer=True if config['render_mode'] == 'rgb_array' else False,
            use_camera_obs=True if config['render_mode'] == 'rgb_array' else False,
            horizon=config['horizon'],
            control_freq=20,
            controller_configs=load_controller_config(default_controller='OSC_POSE'),
            reward_scale=1.0,
            ignore_done=True,
            reward_shaping=config['reward_shaping'],
            camera_names="frontview",
            task_config=config['task_config'] if 'task_config' in config else None,
        )
        super().__init__(GymWrapper(env), max_episode_steps=config['horizon'])
        self.env.render_mode = config['render_mode']
        self._spec = EnvSpec(
            id="TimeLimitLift-v1",
            max_episode_steps=self._max_episode_steps,
        )
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'semantics.async': True,
        }

    @property
    def spec(self):
        return self._spec


class TimeLimitWipe(TimeLimit):
    def __init__(self, config: Dict):
        env = suite.make(
            env_name="Wipe",
            robots="Panda",
            has_renderer= True if config['render_mode'] == 'human' else False,
            has_offscreen_renderer=True if config['render_mode'] == 'rgb_array' else False,
            use_camera_obs=True if config['render_mode'] == 'rgb_array' else False,
            horizon=config['horizon'],
            control_freq=20,
            controller_configs=load_controller_config(default_controller='OSC_POSE'),
            reward_scale=1.0,
            ignore_done=True,
            reward_shaping=config['reward_shaping'],
            camera_names="frontview",
            task_config=config['task_config'] if 'task_config' in config else None,
            hard_reset=False,
        )
        super().__init__(GymWrapper(env), max_episode_steps=config['horizon'])
        self.env.render_mode = config['render_mode']
        self._spec = EnvSpec(
            id="TimeLimitWipe-v1",
            max_episode_steps=self._max_episode_steps,
        )
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'semantics.async': True,
        }

    @property
    def spec(self):
        return self._spec


class TimeLimitPolishing(TimeLimit):
    def __init__(self, config: Dict):
        env = suite.make(
            env_name="Polishing",
            robots="Panda",
            has_renderer= True if config['render_mode'] == 'human' else False,
            has_offscreen_renderer=True if config['render_mode'] == 'rgb_array' else False,
            use_camera_obs=True if config['render_mode'] == 'rgb_array' else False,
            horizon=config['horizon'],
            control_freq=20,
            reward_scale=1.0,
            ignore_done=False,
            reward_shaping=config['reward_shaping'],
            camera_names="frontview",
            task_config=config['task_config'] if 'task_config' in config else None,
            hard_reset=False,
            controller_configs=config['controller_configs'] if 'controller_configs' in config else
                load_controller_config(default_controller='OSC_POSE'),
        )
        super().__init__(GymWrapper(env), max_episode_steps=config['horizon'])
        self.env.render_mode = config['render_mode']
        self._spec = EnvSpec(
            id="TimeLimitPolishing-v1",
            max_episode_steps=self._max_episode_steps,
        )
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'semantics.async': True,
        }

    @property
    def spec(self):
        return self._spec
