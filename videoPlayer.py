import os
import imageio
import pathlib
import numpy as np
import os, shutil

directory="/Users/nardis/Downloads/trajectories"
directory = pathlib.Path(directory).expanduser()

video_dir = "/Users/nardis/Downloads/videos"

for filename in os.listdir(video_dir):
    file_path = os.path.join(video_dir, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

for filename in sorted(directory.glob("*.npz")):
    with filename.open("rb") as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
    print(episode.keys())
    imageio.mimsave("/Users/nardis/Downloads/videos/" + str(len(episode["augmented"])) + "_" + filename.name[:-3] + "mp4", episode["augmented"])
