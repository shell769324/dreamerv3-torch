import pathlib
import numpy as np
import os, shutil
from envs.crafter import stats_keys, inventory_keys

directory="/Users/huizhz/PycharmProjects/dreamerv3-torch/trajectories"
directory = pathlib.Path(directory).expanduser()
obs_keys = ['image', 'augmented', 'reward', 'is_first', 'is_last', 'is_terminal', 'target', 'navigate_target', 'combat_target', 'prev_target', 'prev_combat_target', 'prev_navigate_target', 'actor_mode', 'target_explore_steps', 'target_do_steps', 'distance', 'where', 'reward_mode', 'reward_type', 'target_face_steps', 'target_touch_steps', 'front', 'multi_reward_types', 'log_achievement_collect_coal', 'log_achievement_collect_diamond', 'log_achievement_collect_drink', 'log_achievement_collect_iron', 'log_achievement_collect_sapling', 'log_achievement_collect_stone', 'log_achievement_collect_wood', 'log_achievement_defeat_skeleton', 'log_achievement_defeat_zombie', 'log_achievement_eat_cow', 'log_achievement_eat_plant', 'log_achievement_make_iron_pickaxe', 'log_achievement_make_iron_sword', 'log_achievement_make_stone_pickaxe', 'log_achievement_make_stone_sword', 'log_achievement_make_wood_pickaxe', 'log_achievement_make_wood_sword', 'log_achievement_place_furnace', 'log_achievement_place_plant', 'log_achievement_place_stone', 'log_achievement_place_table', 'log_achievement_wake_up', 'discount', 'objects', 'stats', 'inventory', 'time', 'action', 'logprob']

video_dir = "/Users/huizhz/PycharmProjects/dreamerv3-torch/videos"
action_list = ["noop","move_left","move_right","move_up","move_down","do","sleep","place_stone","place_table",
               "place_furnace","place_plant","make_wood_pickaxe","make_stone_pickaxe","make_iron_pickaxe",
               "make_wood_sword","make_stone_sword","make_iron_sword"]
object_list = ["None","water","grass","stone","path","sand","tree","lava","coal","iron","diamond","table","furnace",
               "player","cow","zombie","skeleton","arrow","plant"]


def print_stats(episode, stats_name, index):
    print("{} = {{".format(stats_name))
    print("  \"position\": (4, 3), # facing + position computes the location of faced object")
    for i, name in enumerate(stats_keys):
        print("  \"{}\": {},".format(name, episode["stats"][index][i]))
    print("  \"facing\": ({}, {}),".format(episode["stats"][index][4], episode["stats"][index][5]))
    print("}")


def print_inventory(episode, inventory_name, index):
    print("{} = {{".format(inventory_name))
    for i, name in enumerate(inventory_keys):
        print("  \"{}\": {},".format(name, episode["inventory"][index][i]))
    print("}")


def print_world(episode, world_name, index):
    print("{} = {{".format(world_name))
    object_map = {}
    for o in object_list:
        object_map[o] = []
    for i in range(episode["objects"][index].shape[0]):
        for j in range(episode["objects"][index].shape[1]):
            object_map[object_list[episode["objects"][index][i][j]]].append((i, j))
    for name in object_list:
        if name not in ["None", "path", "sand", "grass"]:
            print("  \"{}\": {},".format(name, object_map[name]))
    print("}")


def ask_for_goal(prev_goal):
    for filename in sorted(directory.glob("*.npz")):
        with filename.open("rb") as f:
            episode = np.load(f)
            episode = {k: episode[k] for k in episode.keys()}
            for i in range(2, len(episode["action"])):
                print("The previous goal is \"{}\". I've taken a few steps. Please propose a new goal".format(prev_goal))
                print_stats(episode, "player_stats", i)
                print("")
                print_inventory(episode, "inventory", i)
                print("")
                print_world(episode, "world", i)
                print("")
                print("action = {}".format(action_list[np.argmax(episode["action"][i])]))
                print("")
                print("time = {}".format(
                    "morning" if episode["time"][i] > 0.5 else ("afternoon" if episode["time"][i] > 0.3 else "night")))
                return


def ask_for_reward_function():
    for filename in sorted(directory.glob("*.npz")):
        with filename.open("rb") as f:
            episode = np.load(f)
            episode = {k: episode[k] for k in episode.keys()}
            for i in range(2, len(episode["action"])):
                print_stats(episode, "previous_player_stats", i - 1)
                print("")
                print_stats(episode, "current_player_stats", i)
                print("")
                print_inventory(episode, "previous_inventory", i - 1)
                print("")
                print_inventory(episode, "current_inventory", i)
                print("")
                print_world(episode, "previous_world", i - 1)
                print("")
                print_world(episode, "current_world", i)
                print("")
                print("action = \"{}\"".format(action_list[np.argmax(episode["action"][i])]))
                print("")
                print("time = \"{}\"".format("morning" if episode["time"][i] > 0.5 else ("afternoon" if episode["time"][i] > 0.3 else "night")))
                break
            break


ask_for_goal("navigate to tree")
