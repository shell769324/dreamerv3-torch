from tools import SliceDataset
from envs.crafter import targets
import json
import numpy as np
import matplotlib.pyplot as plt
import copy

path = "/Users/nardis/Downloads/trajectories/navigate.json"

analytics = [dict() for _ in range(len(targets))]

with open(path) as json_file:
    json_dict = json.load(json_file)
    tuples = json_dict["tuples"]
    episode_sizes = json_dict["episode_sizes"]
    aggregate_sizes = json_dict["aggregate_sizes"]
    for i in range(len(targets)):
        for ep_name, count in episode_sizes[i].items():
            total = 0
            for j, (st, ed) in enumerate(tuples[i][ep_name]):
                if ed - st not in analytics[i]:
                    analytics[i][ed - st] = 0
                analytics[i][ed - st] += 1

fig = plt.figure()
fig.set_size_inches(7.5, 10)
gs = fig.add_gridspec(len(targets), hspace=0.5)
axs = gs.subplots(sharex=True)
for i in range(len(targets)):
    print(targets[i])
    together = sorted([(k, v) for k, v in analytics[i].items()])
    x = [k for (k, _) in together]
    y = [v for (_, v) in together]
    ay = copy.copy(y)
    for j in range(len(y)):
        ay[j] *= j
        ay[j] += ay[j - 1]if j > 0 else 0
    axs[i].plot(x, ay)
    axs[i].set_title(targets[i])
    stop = 0
    for j in range(len(ay) - 1):
        for k in range(1, 4):
            if ay[j] <= ay[-1] * k/4 <= ay[j + 1]:
                axs[i].plot([x[j], x[j]], [0, ay[-1]], 'k-')
                print(x[j])

plt.show()