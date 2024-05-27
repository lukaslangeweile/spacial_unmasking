import freefield
import pandas
import slab
import os
import pathlib
import time
import numpy
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sounds = {}

def initialize():
    procs = [["RX81", "RX8", DIR / "data" / "rcx" / "cathedral_play_buf.rcx"],
             ["RP2", "RP2", DIR / "data" / "rcx" / "button_numpad.rcx"]]
    freefield.initialize("cathedral", device=procs, zbus=False, connection="USB")
    freefield.SETUP = "cathedral"
    freefield.SPEAKERS = freefield.read_speaker_table()
    freefield.set_logger("DEBUG")
    stim_DIR = DIR / "data" / "stim_files" / "tts-numbers_reversed"
    sound_files = [file for file in stim_DIR.iterdir()]
    for file in sound_files:
        sounds.update({os.path.basename(file): slab.Sound.read(str(file))})
def start_trial(sub_id, n_reps):
    return

def get_sounds_with_filenames(n):
    global sounds
    if n > len(sounds):
        raise ValueError("n cannot be greater than the length of the input list")
    random_indices = np.random.choice(len(sounds), n, replace=False)
    sounds_list = list(sounds.values())
    filenames_list = list(sounds.keys())
    sounds_list = [sounds_list[i] for i in random_indices]
    filenames_list = [filenames_list[i] for i in random_indices]
    return filenames_list, sounds_list