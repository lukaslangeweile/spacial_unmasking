import freefield
import pandas
import slab
import os
import pathlib
import time
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

DIR = DIR = pathlib.Path(os.curdir)

def initialize_setup():
    #initialize for the whole setup, so all 3 experiments can run
    return
def quadratic_func(x, a, b, c):
    return a * x ** 2 + b * x + c

def logarithmic_func(x, a, b, c):
    return a * np.log(b * x) + c

def get_log_parameters(distance):
    parameters_file = DIR / "data" / "mgb_equalization_parameters" / "logarithmic_function_parameters.csv"
    parameters_df = pd.read_csv(parameters_file)
    params = parameters_df[parameters_df['speaker_distance'] == distance]
    a, b, c = params.iloc[0][['a', 'b', 'c']]
    return a, b, c

def get_speaker_normalisation_level(speaker, mgb_loudness=30):
    a, b, c = get_log_parameters(speaker.distance)
    return logarithmic_func(x=mgb_loudness, a=a, b=b, c=c)

def get_stim_dir(stim_type):
    if stim_type == "babble":
        stim_dir =  DIR / "data" / "stim_files" / "babble-numbers-reversed-n13-shifted_resamp_48828"
    elif stim_type == "pinknoise":
        stim_dir = DIR / "data" / "stim_files" / "pinknoise"
    elif stim_type == "countries":
        stim_dir = DIR / "data" / "stim_files" / "tts-countries_n13_resamp_48828"
    elif stim_type == "pinknoise":
        stim_dir = DIR / "data" / "stim_files" / "tts-numbers_n13_resamp_48828"
    else:
        return
    return stim_type

def apply_mgb_equalization(signal, speaker, mgb_loudness=30, fluc=0):
    a, b, c = get_log_parameters(speaker.distance)
    signal.level = logarithmic_func(mgb_loudness + fluc, a, b, c)
    return signal

def record_stimuli_and_create_csv(stim_type="countries"):
    sounds_list, filenames_list = get_sounds_with_filenames(n="all", stim_type="stim_type")
    for i in range(len(sounds_list)):
        for speaker in freefield.SPEAKERS:
            sound = sounds_list[i]
            sound = apply_mgb_equalization(signal=sound, speaker=speaker, mgb_loudness=30)
            rec_sound = freefield.play_and_record(speaker=speaker, sound=sound, compensate_delay=True,
                                                  equalize=False)
            sound_name = os.path.splitext(filenames_list[i])[0]
            filepath = DIR / "data" / "recordings" / " countries" / f"sound-{sound_name}_distance-{speaker.distance}.wav"
            filepath = pathlib.Path(filepath)
            rec_sound.write(filename=filepath, normalise=False)
            time.sleep(2.7)

    #TODO: saving sound data to csv-file

    return

def get_sounds_with_filenames(n="all", stim_type="babble", randomize=False):
    stim_dir = get_stim_dir(stim_type)
    sounds_list = []
    filenames_list = []

    for file in stim_dir.iterdir():
        sounds_list.append(slab.Sound.read(file))
        filenames_list.append(str(file))

    if isinstance(n, int):
        if randomize:
            random_indices = np.random.choice(len(sounds_list), n, replace=False)
            sounds_list = [sounds_list[i] for i in random_indices]
            filenames_list = [filenames_list[i] for i in random_indices]
        else:
            sounds_list = [sounds_list[i] for i in range(n)]
            filenames_list = [filenames_list[i] for i in range(n)]

    return sounds_list, filenames_list

