import slab
import freefield
import pickle
import os
import pathlib
import numpy as np

DIR = os.getcwd()
original_DIR = pathlib.Path(DIR + "/data/stim_files/tts-numbers_n13_resamp_48828")
target_DIR = pathlib.Path(DIR + "/data/stim_files/tts-numbers_reversed")

for file in pathlib.Path.iterdir(original_DIR):
    basename = os.path.splitext(os.path.basename(file))[0]
    sound = slab.Sound.read(file)
    new_data = sound.data[::-1]
    sound.data = new_data
    sound.write(filename=f"{target_DIR}/{basename}_reversed.wav")

