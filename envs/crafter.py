import numpy as np
import gym
import crafter
import itertools

aware = ["water", "stone", "tree", "coal", "iron", "cow", "zombie", "skeleton", "lava"]
target_mapping_temp = ["collect_drink", "collect_stone", "collect_wood", "collect_coal", "collect_iron", "eat_cow",
                       "defeat_zombie", "defeat_skeleton"]
targets = ["water", "stone", "tree", "coal", "iron", "cow", "zombie", "skeleton"]
navigate_targets = ["water", "stone", "tree", "coal", "iron", "cow"]
combat_targets = ["zombie", "skeleton"]
reward_types = {"lava":(-5, 0), "explore_stable":(0, 1), "explore_spot": (1, 2), "navigate_do": (1.5, 3),
                "navigate_face": (0.5, 4), "navigate_lost": (-1.5, 5), "navigate_closer": (0.5, 6),
                "navigate_farther": (-0.5, 7), "navigate_avert": (-0.5, 8), "navigate_stable": (0, 9),
                "combat_do": (1.5, 10), "combat_face": (0.5, 11), "combat_closer": (0.5, 12),
                "combat_zombie_beaten": (-0.5, 13), "combat_arrow": (-0.5, 14), "combat_farther": (-0.5, 15),
                "combat_lost": (0, 16), "combat_stable": (0, 17), "combat_avert": (-0.5, 18), "default": (0, 19)}
actor_mode_map = {"navigate": 0, "explore": 1, "combat": 2}
actor_mode_list = ["navigate", "explore", "combat"]
target_step_list = ["target_do_steps", "target_explore_steps", "target_do_steps"]
target_mode_list = ["navigate_target", "navigate_target", "combat_target"]

directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

reward_type_reverse = [""] * len(reward_types.keys())
for k, (a, b) in reward_types.items():
    reward_type_reverse[b] = k
achievement_mapping = dict()
for i in range(len(targets)):
    achievement_mapping[targets[i]] = target_mapping_temp[i]
    achievement_mapping[target_mapping_temp[i]] = targets[i]



class Crafter():
    def __init__(self, task, size=(64, 64), outdir=None, seed=None):
        assert task in ('reward', 'noreward')
        self._env = crafter.Env(size=size, reward=(task == 'reward'), seed=seed)
        self._crafter_env = self._env
        self._size = size
        self._achievements = crafter.constants.achievements.copy()
        self._done = True
        self.target = np.random.randint(0, len(navigate_targets))
        self.navigate_target = self.target
        self.combat_target = -1
        self.prev_target = np.random.randint(0, len(navigate_targets))
        self.prev_navigate_target = self.target
        self.prev_combat_target = 0
        self._id_to_item = [""] * 19
        self._last_min_dist = None
        self.target_do_steps = 0
        self.target_explore_steps = 0
        for name, ind in itertools.chain(self._env._world._mat_ids.items(), self._env._sem_view._obj_ids.items()):
            name = str(name)[str(name).find('objects.') + len('objects.'):-2].lower() if 'objects.' in str(
                name) else str(name)
            self._id_to_item[ind] = name
        self.actor_mode = 1
        self.prev_actor_mode = 1
        self._row_side = self._env._local_view._grid[0] // 2
        self._col_side = self._env._local_view._grid[1] // 2
        self.value = 0
        self.reward = 0
        self.prev_actual_reward = 0
        self.prev_info = None
        self.reward_type = None
        self.was_facing = False # face in the last step
        self.touched = False
        self.faced = False # has faced once since navigate
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
        # target index with regard to the array that contain both combat and navigate target
        spaces["target"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8)
        spaces["navigate_target"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8)
        spaces["combat_target"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8)
        spaces["where"] = gym.spaces.Box(-np.inf, np.inf, (len(aware), 4), dtype=np.uint8)
        spaces["front"] = gym.spaces.Box(-np.inf, np.inf, (len(aware) + 1,), dtype=np.uint8)
        spaces["distance"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        spaces["target_do_steps"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.int16)
        spaces["target_touch_steps"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.int16)
        spaces["target_face_steps"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.int16)
        spaces["target_explore_steps"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.int16)
        spaces["actor_mode"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8)
        spaces["prev_target"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8)
        spaces["prev_navigate_target"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8)
        spaces["prev_combat_target"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8)
        spaces["reward_mode"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8)
        spaces["reward_type"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8)
        spaces["multi_reward_types"] = gym.spaces.Box(-np.inf, np.inf, (len(reward_types),), dtype=np.uint8)
        spaces.update({
            f'log_achievement_{k}': gym.spaces.Box(-np.inf, np.inf, dtype=np.float32)
            for k in self._achievements})
        return spaces

    @property
    def action_space(self):
        return gym.spaces.Discrete(5)

    def _get_dist(self, player_pos, info):
        min_dist = None
        for i in range(-self._row_side, self._row_side + 1):
            for j in range(-self._col_side, self._col_side + 1):
                x, y = player_pos[0] + i, player_pos[1] + j
                if 0 <= x < self._size[0] and 0 <= y < self._size[1] and self._id_to_item[info['semantic'][x][y]] == \
                        targets[self.target]:
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


    def create_augment(self):
        return self._env.render_target(targets[self.target], self._last_min_dist, self.prev_actual_reward,
                                self.value, self.reward,
                                self.compute_where(self._crafter_env._player.pos,
                                                   self._env._sem_view()),
                                self.predicted_where, self.prev_actor_mode,
                                self.compute_front(self._crafter_env._player.pos,
                                                   self._crafter_env._player.facing,
                                                   self._env._sem_view()))

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
        self.actor_mode = 1
        self.prev_actor_mode = 1
        self.multi_reward_types = np.zeros(len(reward_types), dtype=np.uint8)
        self.target = np.random.randint(0, len(navigate_targets))
        self.navigate_target = self.target
        self.combat_target = 0
        self.target_explore_steps = 0
        self.target_do_steps = 0
        self.faced = False
        self._last_min_dist = self._get_dist(self._crafter_env._player.pos, info)
        where_array = self.compute_where(self._crafter_env._player.pos, info['semantic'])
        front = self.compute_front(self._crafter_env._player.pos, self._crafter_env._player.facing, info['semantic'])
        self.predicted_where = np.zeros((len(aware), 4), dtype=np.uint8)
        self.front = len(aware) + 1
        augmented = self._env.render_target(targets[self.target], self._last_min_dist, 0, self.value, self.reward,
                                            where_array, self.predicted_where, self.prev_actor_mode, front)
        self.prev_actual_reward = 0
        self.touched = False
        self.prev_info = info
        self.was_facing = False
        if self._last_min_dist is None:
            return self.explore_obs(image, 0, info, is_first=True, augmented=augmented, where=where_array, target_explore_steps=0, front=front)
        return self.navigate_obs(image, 0, info, is_first=True, augmented=augmented, where=where_array, front=front)

    def step(self, action):
        self.prev_target = self.target
        self.prev_navigate_target = self.navigate_target
        self.prev_combat_target = self.combat_target
        self.multi_reward_types = np.zeros(len(reward_types), dtype=np.uint8)
        self.actor_mode = None
        if self.reward_type == "navigate":
            res = self.navigate_step(action)
        elif self.reward_type == "explore":
            res = self.explore_step(action)
        elif self.reward_type == "combat":
            res = self.combat_step(action)
        else:
            raise ValueError("impossible")
        print(type(self.prev_navigate_target), type(self.prev_combat_target))
        assert type(self.prev_navigate_target) == type(1), "prev_navigate_target is not int type"
        assert type(self.prev_combat_target) == type(1), "prev_combat_target is not int type"
        self.prev_actor_mode = self.actor_mode
        return res

    def get_facing_object(self, facing=None):
        player_pos = self._env._player.pos
        if facing is None:
            facing = self._env._player.facing
        faced_pos = (player_pos[0] + facing[0], player_pos[1] + facing[1])
        face_in_bound = 0 <= faced_pos[0] < self._size[0] and 0 <= faced_pos[1] < self._size[1]
        if face_in_bound:
            facing_object = self._id_to_item[self._env._sem_view()[faced_pos]]
            if facing_object in targets:
                return facing_object
        return None

    def check_for_combat(self, where_array, player_pos, info):
        for i, target in enumerate(combat_targets):
            if np.sum(where_array[aware.index(target)]) > 0:
                self.target_do_steps = 0
                self.target = targets.index(target)
                self.combat_target = i
                self._last_min_dist = self._get_dist(player_pos, info)
                self.faced = self.get_facing_object() == targets[self.target]
                self.touched = self._last_min_dist == 1
                self.actor_mode = 2
                self.was_facing = self.faced
                break

    def set_for_navigate(self, player_pos, info):
        self.target_do_steps = 0
        self.navigate_target = np.random.randint(0, len(navigate_targets))
        self.target = targets.index(navigate_targets[self.navigate_target])
        self._last_min_dist = self._get_dist(player_pos, info)
        self.actor_mode = 0 if self._last_min_dist is not None else 1
        self.was_facing = self.get_facing_object() == targets[self.target]
        self.touched = self._last_min_dist == 1
        self.faced = self.was_facing

    def navigate_step(self, action):
        if len(action.shape) >= 1:
            action = np.argmax(action)

        assert self.prev_info is not None, "prev info is None"
        # don't do noop
        action += 1

        augmented = self.create_augment()
        image, reward, self._done, info = self._env.step(action)
        where_array = self.compute_where(self._crafter_env._player.pos, self._env._sem_view())
        front = self.compute_front(self._crafter_env._player.pos, self._crafter_env._player.facing, self._env._sem_view())
        self.target_do_steps += 1
        player_pos = info['player_pos']

        reward = np.float32(0)
        # Hit lava very negative reward

        reward_type = "navigate_stable"
        target_do_steps = -1

        achievement = achievement_mapping[navigate_targets[self.navigate_target]]
        touch_step = -1
        face_step = -1
        if self.prev_info['achievements'][achievement] < info['achievements'][achievement]:
            target_do_steps = self.target_do_steps
            self.set_for_navigate(player_pos, info)
            if self.faced:
                face_step = 0
            if self.touched:
                touch_step = 0
            reward_type = "navigate_do"
            reward += reward_types.get(reward_type)[0]
        elif self.get_facing_object() == navigate_targets[self.navigate_target]:
            if not self.was_facing:
                self.was_facing = True
                reward_type = "navigate_face"
                reward += reward_types.get(reward_type)[0]
                if not self.faced:
                    face_step = self.target_do_steps
                if not self.touched:
                    touch_step = self.target_do_steps
                self.faced = True
                self.touched = True
            self._last_min_dist = self._get_dist(player_pos, info)
        else:
            # For measuring distance, we should use previous image since objects may move
            min_dist = self._get_dist(player_pos, info)
            if min_dist == 1:
                if not self.touched:
                    touch_step = self.target_do_steps
                    self.touched = True
            if self._last_min_dist is None:
                raise RuntimeError("Illegal state for navigate, none last min dist")
            elif min_dist is None:
                reward_type = "navigate_lost"
                self.touched = False
                self.faced = False
                self.target_do_steps = 0
                self.actor_mode = 1
            elif self._last_min_dist > min_dist:
                reward_type = "navigate_closer"
            elif self._last_min_dist < min_dist:
                reward_type = "navigate_farther"
            elif self.was_facing:
                reward_type = "navigate_avert"
            else:
                reward_type = "navigate_stable"
            reward += reward_types.get(reward_type)[0]
            self._last_min_dist = min_dist
            self.was_facing = False
        self.check_for_combat(where_array, player_pos, info)
        if self._env._world[player_pos][0] == 'lava':
            reward_type = "lava"
            reward += reward_types.get(reward_type)[0]
        if self.actor_mode is None:
            self.actor_mode = 0 if self._last_min_dist is not None else 1
        self.prev_info = info
        self.prev_actual_reward = reward
        return self.navigate_obs(
            image, reward, info, augmented=augmented,
            is_last=self._done,
            is_terminal=info['discount'] == 0, target_do_steps=target_do_steps,
            where=where_array, reward_type=reward_type, face_step=face_step,
            touch_step=touch_step, front=front), reward, self._done, info

    def navigate_obs(
            self, image, reward, info,
            is_first=False, is_last=False, is_terminal=False, augmented=None,
            target_do_steps=-1, where=None, reward_type="default", face_step=-1, touch_step=-1,
            front=len(aware)):
        assert self.actor_mode is not None, "actor mode in navigate obs is None"
        log_achievements = {
            f'log_achievement_{k}': info['achievements'][k] if info else 0 for k in self._achievements
        }
        self.multi_reward_types[reward_types[reward_type][1]] = 1
        return dict(
            image=image,
            augmented=augmented,
            reward=reward,
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal,
            target=self.target,
            navigate_target=self.navigate_target,
            combat_target=self.combat_target,
            prev_target=self.prev_target,
            prev_combat_target=self.prev_combat_target,
            prev_navigate_target=self.prev_navigate_target,
            actor_mode=self.actor_mode,
            target_explore_steps=-1,
            target_do_steps=target_do_steps,
            distance=-1.0 if self._last_min_dist is None else float(self._last_min_dist),
            where=where,
            reward_mode=0,
            multi_reward_types=self.multi_reward_types,
            reward_type=reward_types.get(reward_type)[1],
            target_face_steps=face_step,
            target_touch_steps=touch_step,
            front=front,
            **log_achievements,
        )

    def explore_step(self, action):
        if len(action.shape) >= 1:
            action = np.argmax(action)
        # don't do noop
        action += 1

        augmented = self.create_augment()
        image, _, self._done, info = self._env.step(action)
        where_array = self.compute_where(self._crafter_env._player.pos, self._env._sem_view())
        front = self.compute_front(self._crafter_env._player.pos, self._crafter_env._player.facing,
                                                               self._env._sem_view())
        self.target_explore_steps += 1
        target_explore_steps = -1
        # reward = np.float32(reward)
        player_pos = info['player_pos']

        reward = np.float32(0)

        self._last_min_dist = self._get_dist(player_pos, info)
        if self._last_min_dist is not None:
            reward_type = "explore_spot"
            reward += reward_types.get(reward_type)[0]
            target_explore_steps = self.target_explore_steps
            self.target_explore_steps = 0
        else:
            reward_type = "explore_stable"
            reward += reward_types.get(reward_type)[0]
        self.check_for_combat(where_array, player_pos, info)
        if self._env._world[player_pos][0] == 'lava':
            reward_type = "lava"
            reward = np.float32(reward_types.get(reward_type)[0])
        if self.actor_mode is None:
            self.actor_mode = 0 if self._last_min_dist is not None else 1
        self.prev_info = info
        self.prev_actual_reward = reward
        return self.explore_obs(
            image, reward, info, augmented=augmented,
            is_last=self._done,
            is_terminal=info['discount'] == 0, target_explore_steps=target_explore_steps,
            where=where_array, reward_type=reward_type, front=front), \
            reward, self._done, info

    def explore_obs(self, image, reward, info,
                    is_first=False, is_last=False, is_terminal=False, augmented=None,
                    target_explore_steps=-1, where=None, reward_type="default", front=len(aware)):
        assert self.actor_mode is not None, "actor mode in explore obs is None"
        log_achievements = {
            f'log_achievement_{k}': info['achievements'][k] if info else 0
            for k in self._achievements}
        self.multi_reward_types[reward_types[reward_type][1]] = 1
        return dict(
            image=image,
            augmented=augmented,
            reward=reward,
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal,
            target=self.target,
            navigate_target=self.navigate_target,
            combat_target=self.combat_target,
            prev_target=self.prev_target,
            prev_combat_target=self.prev_combat_target,
            prev_navigate_target=self.prev_navigate_target,
            actor_mode=self.actor_mode,
            target_explore_steps=target_explore_steps,
            target_do_steps=-1,
            distance=-1.0 if self._last_min_dist is None else float(self._last_min_dist),
            where=where,
            reward_mode=1,
            reward_type=reward_types.get(reward_type)[1],
            target_face_steps=-1,
            target_touch_steps=-1,
            front=front,
            multi_reward_types=self.multi_reward_types,
            **log_achievements,
        )

    def combat_step(self, action):
        if len(action.shape) >= 1:
            action = np.argmax(action)

        assert self.prev_info is not None, "prev info is None"
        # don't do noop
        action += 1

        zombie_do = False
        fail_to_attack_zombie = False
        if self.get_facing_object() == "zombie":
            if action == 5:
                zombie_do = True
            else:
                fail_to_attack_zombie = True
        fail_to_face_zombie = False
        for facing in directions:
            if facing != self._env._player.facing:
                facing_object = self.get_facing_object(facing=facing)
                if "zombie" == facing_object and directions.index(facing) != (action - 1):
                    fail_to_face_zombie = True

        was_near_arrow = False
        prev_health = self._env._player.health
        for facing in directions:
            facing_object = self.get_facing_object(facing=facing)
            if "arrow" == facing_object:
                was_near_arrow = True

        augmented = self.create_augment()
        image, _, self._done, info = self._env.step(action)
        where_array = self.compute_where(self._crafter_env._player.pos, self._env._sem_view())
        front = self.compute_front(self._crafter_env._player.pos, self._crafter_env._player.facing, self._env._sem_view())
        self.target_do_steps += 1
        player_pos = info['player_pos']

        reward = np.float32(0)
        # Hit lava very negative reward

        target_do_steps = -1
        touch_step = -1
        face_step = -1
        is_near_zombie = False
        for facing in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            facing_object = self.get_facing_object(facing=facing)
            if facing_object is not None and "zombie" == facing_object:
                is_near_zombie = True
        target_despawn = np.sum(where_array[aware.index(targets[self.target])]) == 0

        def lose_case():
            nonlocal touch_step, face_step
            self.check_for_combat(where_array, player_pos, info)
            if self.actor_mode is None:
                self.set_for_navigate(player_pos, info)
            if self.touched:
                touch_step = 0
            if self.faced:
                face_step = 0

        achievement = achievement_mapping[combat_targets[self.combat_target]]
        if not is_near_zombie and was_near_arrow and prev_health - self._env._player.health == 2:
            self.multi_reward_types[reward_types["combat_arrow"][1]] = 1
            reward += reward_types.get("combat_arrow")[0]
        if zombie_do or self.prev_info['achievements'][achievement] < info['achievements'][achievement]:
            reward_type = "combat_do"
            reward += reward_types.get(reward_type)[0]
            if self.prev_info['achievements'][achievement] < info['achievements'][achievement]:
                target_do_steps = self.target_do_steps
                self.target_do_steps = 0
                lose_case()
            elif target_despawn:
                lose_case()
            else:
                self.actor_mode = 2
        elif target_despawn:
            lose_case()
            reward_type = "combat_lost"
        elif fail_to_attack_zombie or fail_to_face_zombie:
            reward_type = "combat_zombie_beaten"
            reward += reward_types.get(reward_type)[0]
            self.actor_mode = 2
        elif self.get_facing_object() == combat_targets[self.combat_target]:
            if not self.was_facing:
                self.was_facing = True
                reward_type = "combat_face"
                reward += reward_types.get(reward_type)[0]
                if not self.faced:
                    face_step = self.target_do_steps
                if not self.touched:
                    touch_step = self.target_do_steps
                self.faced = True
                self.touched = True
            else:
                reward_type = "combat_stable"
            self.actor_mode = 2
        else:
            min_dist = self._get_dist(player_pos, info)
            if min_dist == 1:
                if not self.touched:
                    touch_step = self.target_do_steps
                    self.touched = True
            if self._last_min_dist is None:
                raise RuntimeError("Illegal state for combat, none last min dist")
            elif min_dist is None:
                lose_case()
                reward_type = "combat_lost"
            elif self._last_min_dist > min_dist:
                reward_type = "combat_closer"
            elif self._last_min_dist < min_dist:
                reward_type = "combat_farther"
            elif self.was_facing:
                reward_type = "combat_avert"
            else:
                reward_type = "combat_stable"
            if self.actor_mode is None:
                self.actor_mode = 2
            reward += reward_types.get(reward_type)[0]
            self.was_facing = False
        self._last_min_dist = self._get_dist(player_pos, info)
        if self._env._world[player_pos][0] == 'lava':
            reward_type = "lava"
            reward = np.float32(reward_types.get(reward_type)[0])
        self.prev_info = info
        self.prev_actual_reward = reward
        return self.combat_obs(
            image, reward, info, augmented=augmented,
            is_last=self._done,
            is_terminal=info['discount'] == 0, target_do_steps=target_do_steps,
            where=where_array, reward_type=reward_type, face_step=face_step,
            touch_step=touch_step, front=front), reward, self._done, info

    def combat_obs(
            self, image, reward, info,
            is_first=False, is_last=False, is_terminal=False, augmented=None,
            target_do_steps=-1, where=None, reward_type="default", face_step=-1, touch_step=-1,
            front=len(aware)):
        assert self.actor_mode is not None, "actor mode in combat obs is None"
        log_achievements = {
            f'log_achievement_{k}': info['achievements'][k] if info else 0 for k in self._achievements
        }
        self.multi_reward_types[reward_types[reward_type][1]] = 1
        return dict(
            image=image,
            augmented=augmented,
            reward=reward,
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal,
            target=self.target,
            navigate_target=self.navigate_target,
            combat_target=self.combat_target,
            prev_target=self.prev_target,
            prev_combat_target=self.prev_combat_target,
            prev_navigate_target=self.prev_navigate_target,
            actor_mode=self.actor_mode,
            target_explore_steps=-1,
            target_do_steps=target_do_steps,
            distance=-1.0 if self._last_min_dist is None else float(self._last_min_dist),
            where=where,
            reward_mode=2,
            multi_reward_types=self.multi_reward_types,
            reward_type=reward_types.get(reward_type)[1],
            target_face_steps=face_step,
            target_touch_steps=touch_step,
            front=front,
            **log_achievements,
        )

    def render(self):
        return self._env.render()
