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
import json

DIR = DIR = pathlib.Path(os.curdir)

def initialize_setup():
    #initialize for the whole setup, so all 3 experiments can run
    procs = [["RX81", "RX8", DIR / "data" / "rcx" / "cathedral_play_buf.rcx"],
             ["RP2", "RP2", DIR / "data" / "rcx" / "button_numpad.rcx"]]
    freefield.initialize("cathedral", device=procs, zbus=False, connection="USB")
    freefield.SETUP = "cathedral"
    freefield.SPEAKERS = freefield.read_speaker_table()
    freefield.set_logger("DEBUG")
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
    elif stim_type == "syllable":
        stim_dir = DIR / "data" / "stim_files" / "tts-numbers_n13_resamp_48828"
    elif stim_type == "uso":
        stim_dir = DIR / "data" / "stim_files" / "uso_300ms"
    else:
        return
    return pathlib.Path(stim_dir)

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

def get_sounds_dict(stim_type="babble"):
    stim_dir = get_stim_dir(stim_type)
    sounds_dict = {}

    for file in stim_dir.iterdir():
        sounds_dict.update({str(file): slab.Sound.read(file)})

    return sounds_dict

def get_sounds_with_filenames(sounds_dict, n="all", randomize=False):
    filenames_list = sounds_dict.values
    sounds_list = sounds_dict.keys

    if isinstance(n, int):
        if randomize:
            random_indices = np.random.choice(len(sounds_list), n, replace=False)
            sounds_list = [sounds_list[i] for i in random_indices]
            filenames_list = [filenames_list[i] for i in random_indices]
        else:
            sounds_list = [sounds_list[i] for i in range(n)]
            filenames_list = [filenames_list[i] for i in range(n)]

    return sounds_list, filenames_list

def create_localisation_config_file():
    config_dir = DIR / "config"
    filename = config_dir / "localisation_config.json"
    config_data = {"sounds_dict_pinknoise": get_sounds_dict("pinknoise"),
                   "sounds_dict_syllable": get_sounds_dict("syllable"),
                   "sounds_dict_babble": get_sounds_dict("babble")}
    with open(filename, 'w') as config_file:
        json.dump(config_data, config_file, indent=4)


def read_config_file(experiment):
    if experiment == "localisation":
        filename = DIR / "config" / "localisation_config.json"
    elif experiment == "spacial_unmasking":
        filename = DIR / "config" / "spacial-unmasking_config.json"
    elif experiment == "numerosity_judgement":
        filename = DIR / "config" / "numerosity-judgement_config.json"
    else:
        return
    with open(filename, 'r') as config_file:
        config_data = json.load(config_file)
    return config_data

def set_multiple_signals(signals, speakers, equalize=True, mgb_loudness=30, fluc=0):
    for i in range(len(signals)):
        if equalize:
            signals[i] = apply_mgb_equalization(signals[i], speakers[i], mgb_loudness, fluc)
        freefield.set_signal_and_speaker(signals[i], speakers[i], equalize=False)

    for i in range(len(signals)+1, 8):
        freefield.write(tag=f"chan{i}", value=99, processors="RX81")
"""if __name__ == "__main__":"""
