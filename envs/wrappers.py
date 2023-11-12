import gym
import numpy as np
import uuid
import torch
from tools import get_episode_name
from envs.crafter import reward_type_reverse, aware
import json
import os
from tools import thresholds, lava_collect_limit


class CollectDataset:
    def __init__(
        self, env, crafter_env, navigate_dataset, explore_dataset, callbacks=None, precision=32, directory=None,
    ):
        self._env = env
        self.crafter_env = crafter_env
        self._callbacks = callbacks or ()
        self._precision = precision
        self._episode = None
        self.navigate_dataset = navigate_dataset
        self.explore_dataset = explore_dataset
        self.directory = directory
        self.policy = None
        self.agent_state = None


    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs = {k: self._convert(v) for k, v in obs.items()}
        transition = obs.copy()
        if isinstance(action, dict):
            transition.update(action)
        else:
            transition["action"] = action
        transition["reward"] = reward
        transition["discount"] = info.get("discount", np.array(1 - float(done)))
        self._episode[-1]["augmented"] = transition["augmented"]
        self._episode.append(transition)
        if done:
            # Last augmented frame
            if self.policy is not None and self.agent_state is not None:
                results = [(obs, reward)]
                usable_obs, _ = zip(*[p[:] for p in results])
                usable_obs = list(usable_obs)
                usable_obs = {k: np.stack([o[k] for o in usable_obs]) for k in usable_obs[0]}
                self.policy._policy(usable_obs, self.agent_state, True)
                augmented = self.crafter_env.create_augment()
                self._episode[-1]["augmented"] = augmented
            else:
                # this is needed since the dimensions must all match across transition
                self._episode[-1]["augmented"] = self._episode[-2]["augmented"]
            ep_name = str(get_episode_name(self.directory))
            begin = 0
            for i, transition in enumerate(self._episode):
                if i == 0:
                    continue
                if (transition["target_spot"] != self._episode[i - 1]["target_spot"]
                        or transition["target"] != self._episode[i - 1]["target"]):
                    dataset = [self.navigate_dataset, self.explore_dataset][self._episode[i - 1]["target_spot"]]
                    step_name = ["target_navigate_steps", "target_explore_steps"][self._episode[i - 1]["target_spot"]]
                    is_success = transition[step_name] >= 0
                    tuples = dataset.success_tuples if is_success else dataset.failure_tuples
                    episode_sizes = dataset.success_episode_sizes if is_success else dataset.failure_episode_sizes
                    aggregate_sizes = dataset.success_aggregate_sizes if is_success else dataset.failure_aggregate_sizes
                    end = i + 1
                    threshold = thresholds[["navigate", "explore"][self._episode[i - 1]["target_spot"]]]
                    target = transition["prev_target"]
                    if end - begin >= threshold[target]:
                        if ep_name not in tuples[target]:
                            tuples[target][ep_name] = []
                            episode_sizes[target][ep_name] = 0
                        tuples[target][ep_name].append([begin, end])
                        episode_sizes[target][ep_name] += end - begin
                        aggregate_sizes[target] += end - begin
                    begin = i
            dataset = [self.navigate_dataset, self.explore_dataset][self._episode[-1]["reward_mode"]]
            cache = dataset.failure_tuples
            if ep_name not in cache[transition["target"]]:
                cache[transition["target"]][ep_name] = []
                dataset.failure_episode_sizes[transition["target"]][ep_name] = 0
            if reward_type_reverse[self._episode[-1]["reward_type"]] == "lava":
                start = len(self._episode) - (lava_collect_limit - 1)
                for i in range(len(self._episode) - 2, max(-1, len(self._episode) - lava_collect_limit), -1):
                    if np.sum(self._episode[i]["where"][aware.index("lava")]) == 0:
                        start = i + 1
                        break
                dataset.lava_deaths[ep_name] = (start, len(self._episode))


            cache[transition["target"]][ep_name].append([begin, len(self._episode)])
            dataset.failure_episode_sizes[transition["target"]][ep_name] += len(self._episode) - begin
            dataset.failure_aggregate_sizes[transition["target"]] += len(self._episode) - begin
            for key, value in self._episode[1].items():
                if key not in self._episode[0]:
                    self._episode[0][key] = 0 * value
            episode = {k: [t[k] for t in self._episode] for k in self._episode[0]}
            episode = {k: self._convert(v) for k, v in episode.items()}
            info["episode"] = episode
            for callback in self._callbacks:
                callback(episode)
            self.navigate_dataset.save()
            self.explore_dataset.save()
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        transition = obs.copy()
        # Missing keys will be filled with a zeroed out version of the first
        # transition, because we do not know what action information the agent will
        # pass yet.
        transition["reward"] = 0.0
        transition["discount"] = 1.0
        self._episode = [transition]
        return obs

    def _convert(self, value):
        value = np.array(value)
        if np.issubdtype(value.dtype, np.floating):
            dtype = {16: np.float16, 32: np.float32, 64: np.float64}[self._precision]
        elif np.issubdtype(value.dtype, np.signedinteger):
            dtype = {16: np.int16, 32: np.int32, 64: np.int64}[self._precision]
        elif np.issubdtype(value.dtype, np.uint8):
            dtype = np.uint8
        elif np.issubdtype(value.dtype, np.bool_):
            dtype = np.bool_
        else:
            raise NotImplementedError(value.dtype)
        return value.astype(dtype)


class TimeLimit:
    def __init__(self, env, duration):
        self._env = env
        self._duration = duration
        self._step = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        assert self._step is not None, "Must reset environment."
        obs, reward, done, info = self._env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            if "discount" not in info:
                info["discount"] = np.array(1.0).astype(np.float32)
            self._step = None
        return obs, reward, done, info

    def reset(self):
        self._step = 0
        return self._env.reset()


class NormalizeActions:
    def __init__(self, env):
        self._env = env
        self._mask = np.logical_and(
            np.isfinite(env.action_space.low), np.isfinite(env.action_space.high)
        )
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def action_space(self):
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        return gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        return self._env.step(original)


class OneHotAction:
    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete)
        self._env = env
        self._random = np.random.RandomState()

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def action_space(self):
        shape = (self._env.action_space.n,)
        space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        space.sample = self._sample_action
        space.discrete = True
        return space

    def step(self, action):
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        if not np.allclose(reference, action):
            raise ValueError(f"Invalid one-hot action:\n{action}")
        return self._env.step(index)

    def reset(self):
        return self._env.reset()

    def _sample_action(self):
        actions = self._env.action_space.n
        index = self._random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference


class RewardObs:
    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def observation_space(self):
        spaces = self._env.observation_space.spaces
        assert "reward" not in spaces
        spaces["reward"] = gym.spaces.Box(-np.inf, np.inf, dtype=np.float32)
        return gym.spaces.Dict(spaces)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs["reward"] = reward
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        obs["reward"] = 0.0
        return obs


class SelectAction:
    def __init__(self, env, key):
        self._env = env
        self._key = key

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        return self._env.step(action[self._key])
