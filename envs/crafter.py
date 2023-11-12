import numpy as np
import gym
import crafter
import itertools

targets = ["water", "stone", "tree", "coal", "iron", "cow"]
aware = ["water", "stone", "tree", "coal", "iron", "cow", "lava", "zombie", "skeleton"]
target_mapping_temp = ["collect_drink", "collect_stone", "collect_wood", "collect_coal", "collect_iron", "eat_cow"]
reward_types = {"lava":(-5, 0), "explore_stable":(0, 1), "explore_spot": (1, 2), "navigate_do": (1.5, 3),
                "navigate_face": (0.5, 4), "navigate_lost": (-1.5, 5), "navigate_closer": (0.5, 6),
                "navigate_farther": (-0.5, 7), "navigate_avert": (-0.5, 8), "navigate_stable": (0, 9),
                "default": (0, 10)}

reward_type_reverse = [""] * len(reward_types.keys())
for k, (a, b) in reward_types.items():
    reward_type_reverse[b] = k
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
        self._id_to_item = [""] * 19
        self._last_min_dist = None
        self.target_navigate_steps = 0
        self.target_explore_steps = 0
        for name, ind in itertools.chain(self._env._world._mat_ids.items(), self._env._sem_view._obj_ids.items()):
            name = str(name)[str(name).find('objects.') + len('objects.'):-2].lower() if 'objects.' in str(
                name) else str(name)
            self._id_to_item[ind] = name
        self._row_side = self._env._local_view._grid[0] // 2
        self._col_side = self._env._local_view._grid[1] // 2
        self.value = 0
        self.reward = 0
        self.prev_actual_reward = 0
        self.prev_info = None
        self.reward_type = None
        self.was_facing = False
        self.touched = False
        self.faced = False
        self.predicted_where = np.zeros((len(aware), 4), dtype=np.uint8)
        self.front = len(aware)
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
        spaces["where"] = gym.spaces.Box(-np.inf, np.inf, (len(aware), 4), dtype=np.uint8)
        spaces["front"] = gym.spaces.Box(-np.inf, np.inf, (len(aware) + 1,), dtype=np.uint8)
        spaces["distance"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        spaces["target_navigate_steps"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.int16)
        spaces["target_touch_steps"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.int16)
        spaces["target_face_steps"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.int16)
        spaces["target_explore_steps"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.int16)
        spaces["target_spot"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8)
        spaces["prev_target"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8)
        spaces["reward_mode"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8)
        spaces["reward_type"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8)
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

    def compute_front(self, player_pos, facing, sem):
        faced_pos = (player_pos[0] + facing[0], player_pos[1] + facing[1])
        face_in_bound = 0 <= faced_pos[0] < self._size[0] and 0 <= faced_pos[1] < self._size[1]
        if face_in_bound:
            name = self._id_to_item[sem[faced_pos]]
            if name in aware:
                return aware.index(name)
        return len(aware)

    def compute_where(self, player_pos, sem):
        where = np.zeros((len(aware), 4), dtype=np.uint8)
        def condition(i1, i2, t):
            return 0 <= i1 < len(sem) and 0 <= i2 < len(sem[0]) and self._id_to_item[sem[i1][i2]] == t

        for index, t in enumerate(aware):
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
        return where

    def reset(self):
        self._done = False
        image = self._env.reset()
        achievements = dict()
        for achievement in self._achievements:
            achievements[achievement] = 0
        info = {
            'semantic': self._crafter_env._sem_view(),
            'achievements': achievements,
            'player_pos': self._crafter_env._player.pos,
            'player_facing': self._crafter_env._player.facing,
        }
        self._target = np.random.randint(0, len(targets))
        self.target_explore_steps = 0
        self.target_navigate_steps = 0
        self.faced = False
        self._last_min_dist = self._get_dist(self._crafter_env._player.pos, info)
        where_array = self.compute_where(self._crafter_env._player.pos, info['semantic'])
        front = self.compute_front(self._crafter_env._player.pos, self._crafter_env._player.facing, info['semantic'])
        self.predicted_where = np.zeros((len(aware), 4), dtype=np.uint8)
        self.front = len(aware) + 1
        augmented = self._env.render_target(targets[self._target], self._last_min_dist, 0, self.value, self.reward,
                                            where_array, self.predicted_where, self._last_min_dist is not None, front)
        self.prev_actual_reward = 0
        self.touched = False
        self.prev_info = info
        self.was_facing = False
        if self._last_min_dist is None:
            return self.explore_obs(image, 0, info, is_first=True, augmented=augmented, where=where_array, target_explore_steps=0, front=front)
        return self.navigate_obs(image, 0, info, is_first=True, augmented=augmented, where=where_array, front=front)

    def step(self, action):
        if self.reward_type == "navigate":
            res = self.navigate_step(action)
        elif self.reward_type == "explore":
            res = self.explore_step(action)
        else:
            raise ValueError("impossible")
        return res

    def navigate_step(self, action):
        if len(action.shape) >= 1:
            action = np.argmax(action)

        assert self.prev_info is not None, "prev info is None"
        # don't do noop
        action += 1
        previous_pos = self._crafter_env._player.pos
        cow_do = False
        useless_do = False
        if action == 5:
            player_pos = self.prev_info['player_pos']
            facing = self.prev_info['player_facing']
            faced_pos = (player_pos[0] + facing[0], player_pos[1] + facing[1])
            face_in_bound = 0 <= faced_pos[0] < self._size[0] and 0 <= faced_pos[1] < self._size[1]
            if face_in_bound and self._id_to_item[self.prev_info['semantic'][faced_pos]] == targets[self._target] and self._target == targets.index("cow"):
                cow_do = True
            # if face_in_bound and self._id_to_item[self.prev_info['semantic'][faced_pos]] in ["grass", "path", "sand"]:
            #    useless_do = True

        augmented = self.create_augment()
        image, reward, self._done, info = self._env.step(action)
        where_array = self.compute_where(self._crafter_env._player.pos, self._env._sem_view())
        front = self.compute_front(self._crafter_env._player.pos, self._crafter_env._player.facing, self._env._sem_view())
        self.target_navigate_steps += 1
        # reward = np.float32(reward)
        player_pos = info['player_pos']
        facing = info['player_facing']

        reward = np.float32(0)
        # Hit lava very negative reward

        reward_type = "navigate_stable"
        if self._env._world[player_pos][0] == 'lava':
            reward_type = "lava"
            reward += reward_types.get(reward_type)[0]
        faced_pos = (player_pos[0] + facing[0], player_pos[1] + facing[1])
        target_navigate_steps = -1
        prev_target = self._target
        face_in_bound = 0 <= faced_pos[0] < self._size[0] and 0 <= faced_pos[1] < self._size[1]

        achievement = target_mapping[targets[self._target]]
        touch_step = -1
        face_step = -1
        if reward_type != "lava":
            if self.prev_info['achievements'][achievement] < info['achievements'][achievement]:
                self._target = np.random.randint(0, len(targets))
                self._last_min_dist = self._get_dist(player_pos, info)
                target_navigate_steps = self.target_navigate_steps
                self.target_navigate_steps = 0
                self.was_facing = False
                self.faced = face_in_bound and self._id_to_item[info['semantic'][faced_pos]] == targets[self._target]
                if self.faced:
                    face_step = 0
                self.touched = self._last_min_dist == 1
                if self.touched:
                    touch_step = 0
                reward_type = "navigate_do"
                reward += reward_types.get(reward_type)[0]
            elif cow_do:
                reward_type = "navigate_do"
                reward += reward_types.get(reward_type)[0]
            elif face_in_bound and self._id_to_item[info['semantic'][faced_pos]] == targets[self._target]:
                if not self.was_facing:
                    self.was_facing = True
                    reward_type = "navigate_face"
                    reward += reward_types.get(reward_type)[0]
                    if not self.faced:
                        face_step = self.target_navigate_steps
                    if not self.touched:
                        touch_step = self.target_navigate_steps
                    self.faced = True
                    self.touched = True
            else:
                # For measuring distance, we should use previous image since objects may move
                min_dist = self._get_dist(player_pos, info)
                if min_dist == 1:
                    if not self.touched:
                        touch_step = self.target_navigate_steps
                        self.touched = True
                if self._last_min_dist is None:
                    raise RuntimeError("Illegal state, none last min dist")
                elif min_dist is None:
                    reward_type = "navigate_lost"
                    self.touched = False
                    self.faced = False
                    self.target_navigate_steps = 0
                elif self._last_min_dist > min_dist:
                    reward_type = "navigate_closer"
                elif self._last_min_dist < min_dist:
                    reward_type = "navigate_farther"
                elif self.was_facing:
                    reward_type = "navigate_avert"
                elif useless_do:
                    reward_type = "useless_do"
                else:
                    reward_type = "navigate_stable"
                reward += reward_types.get(reward_type)[0]
                self._last_min_dist = self._get_dist(player_pos, info)
                self.was_facing = False
        self.prev_info = info
        self.prev_actual_reward = reward

        return self.navigate_obs(
            image, reward, info, augmented=augmented,
            is_last=self._done,
            is_terminal=info['discount'] == 0, target_navigate_steps=target_navigate_steps,
            prev_target=prev_target, where=where_array, reward_type=reward_type, face_step=face_step,
            touch_step=touch_step, front=front), reward, self._done, info

    def navigate_obs(
            self, image, reward, info,
            is_first=False, is_last=False, is_terminal=False, augmented=None,
            target_navigate_steps=-1, prev_target=None, where=None, reward_type="default", face_step=-1, touch_step=-1,
            front=len(aware)):
        if prev_target is None:
            prev_target = self._target
        log_achievements = {
            f'log_achievement_{k}': info['achievements'][k] if info else 0 for k in self._achievements
        }
        return dict(
            image=image,
            augmented=augmented,
            reward=reward,
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal,
            target=self._target,
            target_spot=0 if self._last_min_dist is not None else 1,
            target_explore_steps=-1,
            target_navigate_steps=target_navigate_steps,
            prev_target=prev_target,
            distance=-1.0 if self._last_min_dist is None else float(self._last_min_dist),
            where=where,
            reward_mode=0,
            reward_type=reward_types.get(reward_type)[1],
            target_face_steps=face_step,
            target_touch_steps=touch_step,
            front=front,
            **log_achievements,
        )

    def create_augment(self):
        return self._env.render_target(targets[self._target], self._last_min_dist, self.prev_actual_reward,
                                self.value, self.reward,
                                self.compute_where(self._crafter_env._player.pos,
                                                   self._env._sem_view()),
                                self.predicted_where, self._last_min_dist is not None,
                                self.compute_front(self._crafter_env._player.pos,
                                                   self._crafter_env._player.facing,
                                                   self._env._sem_view()))

    def explore_step(self, action):
        if len(action.shape) >= 1:
            action = np.argmax(action)
        # don't do noop
        action += 1

        augmented = self.create_augment()
        useless_do = False
        if action == 5:
            player_pos = self.prev_info['player_pos']
            facing = self.prev_info['player_facing']
            faced_pos = (player_pos[0] + facing[0], player_pos[1] + facing[1])
            face_in_bound = 0 <= faced_pos[0] < self._size[0] and 0 <= faced_pos[1] < self._size[1]
            #if face_in_bound and self._id_to_item[self.prev_info['semantic'][faced_pos]] in ["grass", "path", "sand"]:
            #    useless_do = True
        image, _, self._done, info = self._env.step(action)
        where_array = self.compute_where(self._crafter_env._player.pos, self._env._sem_view())
        front = self.compute_front(self._crafter_env._player.pos, self._crafter_env._player.facing,
                                                               self._env._sem_view())
        self.target_explore_steps += 1
        target_explore_steps = -1
        # reward = np.float32(reward)
        player_pos = info['player_pos']

        reward = np.float32(0)
        # Hit lava very negative reward
        reward_type = "explore_stable"
        if self._env._world[player_pos][0] == 'lava':
            reward_type = "lava"
            reward += reward_types.get(reward_type)[0]
        prev_target = self._target

        self._last_min_dist = self._get_dist(player_pos, info)
        if reward_type != "lava":
            if self._last_min_dist is not None:
                reward_type = "explore_spot"
                reward += reward_types.get(reward_type)[0]
                target_explore_steps = self.target_explore_steps
                self.target_explore_steps = 0
            elif useless_do:
                reward_type = "useless_do"
                reward += reward_types.get(reward_type)[0]
            else:
                reward_type = "explore_stable"
                reward += reward_types.get(reward_type)[0]
        self.prev_info = info
        self.prev_actual_reward = reward
        return self.explore_obs(
            image, reward, info, augmented=augmented,
            is_last=self._done,
            is_terminal=info['discount'] == 0, target_explore_steps=target_explore_steps,
            prev_target=prev_target, where=where_array, reward_type=reward_type, front=front), reward, self._done, info

    def explore_obs(self, image, reward, info,
                    is_first=False, is_last=False, is_terminal=False, augmented=None,
                    target_explore_steps=-1, prev_target=None, where=None, reward_type="default", front=len(aware)):
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
            target_spot=0 if self._last_min_dist is not None else 1,
            target_explore_steps=target_explore_steps,
            target_navigate_steps=-1,
            prev_target=prev_target,
            distance=-1.0 if self._last_min_dist is None else float(self._last_min_dist),
            where=where,
            reward_mode=1,
            reward_type=reward_types.get(reward_type)[1],
            target_face_steps=-1,
            target_touch_steps=-1,
            front=front,
            **log_achievements,
        )

    def render(self):
        return self._env.render()
