import pathlib
import os
import slab

DIR = os.getcwd()
target_DIR = pathlib.Path(DIR + "/data/stim_files/pinknoise")

for i in range(10):
    pinknoise = slab.Sound.pinknoise(duration=1, samplerate=48828)
    pinknoise.write(target_DIR / f"pinknoise_{i}.wav")