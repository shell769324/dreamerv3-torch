import numpy as np
import gym
import crafter
import itertools

targets = ["water", "stone", "tree", "coal", "iron", "cow"]
target_mapping_temp = ["collect_drink", "collect_stone", "collect_wood", "collect_coal", "collect_iron", "eat_cow"]
target_mapping = dict()
for i in range(len(targets)):
    target_mapping[targets[i]] = target_mapping_temp[i]
    target_mapping[target_mapping_temp[i]] = targets[i]


class Crafter():
    def __init__(self, task, size=(64, 64), outdir=None, seed=None):
        assert task in ('reward', 'noreward')
        self._env = crafter.Env(size=size, reward=(task == 'reward'), seed=seed)
        self._crafter_env = self._env
        self._size = size
        self._achievements = crafter.constants.achievements.copy()
        self._done = True
        self._target = np.random.randint(0, len(targets))
        self._id_to_item = [0] * 19
        self._last_min_dist = None
        self.target_reached_steps = 0
        self.target_spot_steps = 0
        for name, ind in itertools.chain(self._env._world._mat_ids.items(), self._env._sem_view._obj_ids.items()):
            name = str(name)[str(name).find('objects.') + len('objects.'):-2].lower() if 'objects.' in str(
                name) else str(name)
            self._id_to_item[ind] = name
        self._row_side = self._env._local_view._grid[0] // 2
        self._col_side = self._env._local_view._grid[1] // 2
        self.value = 0
        self.reward = 0
        self.prev_info = None
        self.reward_type = None
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
        spaces["is_first"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8)
        spaces["is_last"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8)
        spaces["is_terminal"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8)
        spaces["reward"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        spaces["target"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8)
        spaces["where"] = gym.spaces.Box(-np.inf, np.inf, (len(targets) * 4,), dtype=np.uint8)
        spaces["distance"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        spaces["target_reached_steps"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint16)
        spaces["target_spot_steps"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint16)
        spaces["target_reached"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8)
        spaces["target_spot"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8)
        spaces["prev_target"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8)
        spaces["reward_mode"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8)
        spaces.update({
            f'log_achievement_{k}': gym.spaces.Box(-np.inf, np.inf, dtype=np.float32)
            for k in self._achievements})
        return spaces

    @property
    def action_space(self):
        return gym.spaces.Discrete(5)

    def _get_dist(self, player_pos, info, center=None):
        if center is None:
            center = player_pos
        min_dist = None
        for i in range(-self._row_side, self._row_side + 1):
            for j in range(-self._col_side, self._col_side + 1):
                x, y = center[0] + i, center[1] + j
                if 0 <= x < self._size[0] and 0 <= y < self._size[1] and self._id_to_item[info['semantic'][x][y]] == \
                        targets[self._target]:
                    dist = abs(x - player_pos[0]) + abs(y - player_pos[1])
                    min_dist = dist if min_dist is None else min(dist, min_dist)
        return min_dist

    def compute_where(self, player_pos, sem):
        where = np.zeros((len(targets), 4), dtype=np.uint8)

        def condition(i1, i2, t):
            return 0 <= i1 < len(sem) and 0 <= i2 < len(sem[0]) and self._id_to_item[sem[i1][i2]] == t

        for index, t in enumerate(targets):
            lower_row = player_pos[0] - self._row_side
            lower_col = player_pos[1] - self._col_side
            high_row = player_pos[0] + self._row_side + 1
            high_col = player_pos[1] + self._col_side + 1
            for i in range(lower_row, player_pos[0] + 1):
                for j in range(lower_col, player_pos[1] + 1):
                    if condition(i, j, t):
                        where[index][0] = 1
            for i in range(player_pos[0], high_row):
                for j in range(lower_col, player_pos[1] + 1):
                    if condition(i, j, t):
                        where[index][1] = 1
            for i in range(lower_row, player_pos[0] + 1):
                for j in range(player_pos[1], high_col):
                    if condition(i, j, t):
                        where[index][2] = 1
            for i in range(player_pos[0], high_row):
                for j in range(player_pos[1], high_col):
                    if condition(i, j, t):
                        where[index][3] = 1
        return where.reshape(-1)

    def reset(self):
        self._done = False
        image = self._env.reset()
        achievements = dict()
        for achievement in self._achievements:
            achievements[achievement] = 0
        info = {
            'semantic': self._crafter_env._sem_view(),
            'achievements': achievements
        }
        self._target = np.random.randint(0, len(targets))
        self.target_spot_steps = 0
        self.target_reached_steps = 0
        self._last_min_dist = self._get_dist(self._crafter_env._player.pos, info)
        where_array = self.compute_where(self._crafter_env._player.pos, info)
        augmented = self._env.render_target(targets[self._target], self._last_min_dist, 0, self.value, self.reward,
                                            where_array)
        self.prev_info = info
        if self._last_min_dist is None:
            return self.explore_obs(image, 0, info, is_first=True, augmented=augmented, where=where_array)
        return self.navigate_obs(image, 0.0, {}, is_first=True, augmented=augmented, where=where_array)

    def step(self, action):
        if self.reward_type == "navigate":
            return self.navigate_step(action)
        elif self.reward_type == "explore":
            return self.explore_step(action)
        else:
            raise ValueError("impossible")

    def navigate_step(self, action):
        if len(action.shape) >= 1:
            action = np.argmax(action)
        # don't do noop
        action += 1
        previous_pos = self._crafter_env._player.pos

        where_array = self.compute_where(previous_pos, self._env._sem_view())
        image, reward, self._done, info = self._env.step(action)
        self.target_reached_steps += 1
        # reward = np.float32(reward)
        player_pos = info['player_pos']
        facing = info['player_facing']

        reward = np.float32(0)
        # Hit lava very negative reward
        if self._env._world[player_pos][0] == 'lava':
            reward -= 5
        faced_pos = (player_pos[0] + facing[0], player_pos[1] + facing[1])
        target_reached = False
        target_reached_steps = self.target_reached_steps
        prev_target = self._target
        face_in_bound = 0 <= faced_pos[0] < self._size[0] and 0 <= faced_pos[1] < self._size[1]
        if self.prev_info is None:
            self.prev_info = info

        achievement = target_mapping[targets[self._target]]
        if self.prev_info['achievements'][achievement] < info['achievements'][achievement]:
            reward += 1
            self._target = np.random.randint(0, len(targets))
            self._last_min_dist = self._get_dist(player_pos, info)
            target_reached = True
            self.target_reached_steps = 0
        elif face_in_bound and self._id_to_item[info['semantic'][faced_pos]] == targets[self._target]:
            reward += 1
        else:
            # For measuring distance, we should use previous image since objects may move
            delayed_min_dist = self._get_dist(player_pos, self.prev_info, center=previous_pos)
            min_dist = self._get_dist(player_pos, info)
            if self._last_min_dist is None:
                raise RuntimeError("Illegal state, none last min dist")
            elif min_dist is None:
                # Lost track bigger penalty
                reward -= 1
            elif self._last_min_dist > delayed_min_dist:
                reward += 0.5
            elif self._last_min_dist < delayed_min_dist:
                reward -= 0.5
            self._last_min_dist = self._get_dist(player_pos, info)
        augmented = self._env.render_target(targets[self._target], self._last_min_dist, reward, self.value, self.reward,
                                            where_array)
        self.prev_info = info

        return self.navigate_obs(
            image, reward, info, augmented=augmented,
            is_last=self._done,
            is_terminal=info['discount'] == 0, target_reached=target_reached, target_reached_steps=target_reached_steps,
            prev_target=prev_target, where=where_array), reward, self._done, info

    def navigate_obs(
            self, image, reward, info,
            is_first=False, is_last=False, is_terminal=False, augmented=None, target_reached=False,
            target_reached_steps=0, prev_target=None, where=None):
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
            target_spot_steps=0,
            target_spot=self._last_min_dist is not None,
            target_reached_steps=target_reached_steps,
            target_reached=target_reached,
            prev_target=prev_target,
            distance=-1.0 if self._last_min_dist is None else float(self._last_min_dist),
            where=where,
            reward_mode=0,
            **log_achievements,
        )

    def explore_step(self, action):
        if len(action.shape) >= 1:
            action = np.argmax(action)
        # don't do noop
        action += 1
        previous_pos = self._crafter_env._player.pos

        where_array = self.compute_where(previous_pos, self._env._sem_view())
        image, _, self._done, info = self._env.step(action)
        self.target_spot_steps += 1
        target_spot_steps = self.target_spot_steps
        # reward = np.float32(reward)
        player_pos = info['player_pos']

        reward = np.float32(0)
        # Hit lava very negative reward
        if self._env._world[player_pos][0] == 'lava':
            reward -= 5
        prev_target = self._target

        self._last_min_dist = self._get_dist(player_pos, info)
        target_spot = False
        if self._last_min_dist is not None:
            reward += 1
            target_spot = True
            self.target_spot_steps = 0
        self.prev_info = info
        augmented = self._env.render_target(targets[self._target], self._last_min_dist, reward, self.value, self.reward,
                                            where_array)
        return self.explore_obs(
            image, reward, info, augmented=augmented,
            is_last=self._done,
            is_terminal=info['discount'] == 0, target_spot_steps=target_spot_steps,
            prev_target=prev_target, where=where_array), reward, self._done, info

    def explore_obs(self, image, reward, info,
                    is_first=False, is_last=False, is_terminal=False, augmented=None,
                    target_spot_steps=0, prev_target=None, where=None):
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
            target_spot_steps=target_spot_steps,
            target_spot=self._last_min_dist is not None,
            target_reached_steps=0,
            target_reached=False,
            prev_target=prev_target,
            distance=-1.0 if self._last_min_dist is None else float(self._last_min_dist),
            where=where,
            reward_mode=1,
            **log_achievements,
        )

    def render(self):
        return self._env.render()
