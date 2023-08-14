import os
import imageio
import pathlib
import numpy as np

directory="/Users/nardis/Downloads/trajectories"
directory = pathlib.Path(directory).expanduser()

for filename in sorted(directory.glob("*.npz")):
    with filename.open("rb") as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
    print(episode.keys())
    imageio.mimsave("/Users/nardis/Downloads/videos/" + filename.name[:-3] + ".mp4", episode["augmented"])