import random

import numpy as np
import gym
import random
import crafter
import itertools

targets = ["water", "stone", "tree", "coal", "iron", "cow", "skeleton"]

class Crafter():

  def __init__(self, task, size=(64, 64), outdir=None, seed=None):
    assert task in ('reward', 'noreward')
    self._env = crafter.Env(size=size, reward=(task == 'reward'), seed=seed)
    self._crafter_env = self._env
    self._size = size
    self._achievements = crafter.constants.achievements.copy()
    self._done = True
    self._target = np.random.randint(0, len(targets))
    self._step_for_target = 0
    self._id_to_item = [0] * 19
    self._last_min_dist = None
    self._target_steps = 0
    for name, ind in itertools.chain(self._env._world._mat_ids.items(), self._env._sem_view._obj_ids.items()):
        name = str(name)[str(name).find('objects.') + len('objects.'):-2].lower() if 'objects.' in str(name) else str(
            name)
        self._id_to_item[ind] = name
    self._row_side = self._env._local_view._grid[0] // 2
    self._col_side = self._env._local_view._grid[1] // 2
    if outdir:
      self._env = crafter.Recorder(
          self._env, outdir,
          save_stats=True,
          save_video=False,
          save_episode=False,
      )

  @property
  def observation_space(self):
    spaces = {}
    spaces["image"] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
    spaces["augmented"] = gym.spaces.Box(0, 255, self._crafter_env.aug_size + (3,), dtype=np.uint8)
    spaces["log_reward"] = gym.spaces.Box(-np.inf, np.inf, dtype=np.float32)
    spaces["is_first"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8)
    spaces["is_last"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8)
    spaces["is_terminal"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8)
    spaces["reward"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
    spaces["target"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8)
    spaces["target_steps"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint16)
    spaces["target_reached"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8)
    spaces["prev_target"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8)
    spaces.update({
        f'log_achievement_{k}': gym.spaces.Box(-np.inf, np.inf, dtype=np.float32)
        for k in self._achievements})
    return spaces

  @property
  def action_space(self):
    return self._env.action_space

  def _get_dist(self, player_pos, info):
    min_dist = None
    for i in range(-self._row_side, self._row_side + 1):
        for j in range(-self._col_side, self._col_side + 1):
            x, y = player_pos[0] + i, player_pos[1] + j
            if 0 <= x < self._size[0] and 0 <= y < self._size[1] and self._id_to_item[info['semantic'][x][y]] == targets[self._target]:
                dist = abs(i) + abs(j)
                min_dist = dist if min_dist is None else min(dist, min_dist)
    return min_dist

  def reset(self):
    self._done = False
    image = self._env.reset()
    self._target = np.random.randint(0, len(targets))
    self._target_steps = 0
    info = {
        'semantic': self._crafter_env._sem_view()
    }
    self._last_min_dist = self._get_dist(self._crafter_env._player.pos, info)
    augmented = self._env.render_target(targets[self._target], self._last_min_dist, 0)
    return self._obs(image, 0.0, {}, is_first=True, augmented=augmented)

  def step(self, action):
    if len(action.shape) >= 1:
        action = np.argmax(action)
    image, reward, self._done, info = self._env.step(action)
    self._target_steps += 1
    #reward = np.float32(reward)
    reward = np.float32(0)
    player_pos = info['player_pos']
    facing = info['player_facing']
    faced_pos = (player_pos[0] + facing[0], player_pos[1] + facing[1])
    target_reached = False
    target_steps = self._target_steps
    prev_target = self._target
    face_in_bound = 0 <= faced_pos[0] < self._size[0] and 0 <= faced_pos[1] < self._size[1]
    if face_in_bound and self._id_to_item[info['semantic'][faced_pos]] == targets[self._target]:
        reward += 1
        self._target = np.random.randint(0, len(targets))
        self._last_min_dist = self._get_dist(player_pos, info)
        target_reached = True
        self._target_steps = 0
    else:
        if face_in_bound and self._id_to_item[info['semantic'][faced_pos]] in targets:
            reward -= 0.1
        min_dist = self._get_dist(player_pos, info)
        if self._last_min_dist is None:
            if min_dist is not None:
                reward += 0.75
        elif min_dist is None:
            reward -= 0.75
        elif self._last_min_dist > min_dist:
            reward += 0.5
        elif self._last_min_dist < min_dist:
            reward -= 0.5
        self._last_min_dist = min_dist

    augmented = self._env.render_target(targets[self._target], self._last_min_dist, reward)

    return self._obs(
        image, reward, info, augmented=augmented,
        is_last=self._done,
        is_terminal=info['discount'] == 0, target_reached=target_reached, target_steps=target_steps,
        prev_target=prev_target), reward, self._done, info

  def _obs(
      self, image, reward, info,
      is_first=False, is_last=False, is_terminal=False, augmented=None, target_reached=False,
          target_steps=0, prev_target=None):
    if prev_target is None:
        prev_target = self._target
    log_achievements = {
        f'log_achievement_{k}': info['achievements'][k] if info else 0
        for k in self._achievements}
    return dict(
        image=image,
        augmented=augmented,
        reward=reward,
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
        target=self._target,
        log_reward=np.float32(info['reward'] if info else 0.0),
        target_steps=target_steps,
        target_reached=target_reached,
        prev_target=prev_target,
        **log_achievements,
    )

  def render(self):
    return self._env.render()
