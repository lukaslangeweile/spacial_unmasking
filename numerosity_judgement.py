import spacial_unmasking
import freefield
import slab
import pathlib
import os
import seaborn as sns
import numpy as np
import time
import pandas as pd
import statistics


normalisation_method = None
event_id = 0
DIR = pathlib.Path(os.curdir)
num_dict = {"one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six":  6,
            "seven": 7,
            "eight": 8,
            "nine": 9}
target_dict = {}
talkers = ["p229", "p245", "p248", "p256", "p268", "p284", "p307", "p318"]
n_sounds = [1, 2, 3, 4, 5, 6]
SOUND_TYPE = None
sounds = {}

def initialize_setup(normalisation_algorithm="rms", normalisation_sound_type="syllable", sound_type="babble"):
    global normalisation_method
    global sounds
    global SOUND_TYPE
    SOUND_TYPE = sound_type
    procs = [["RX81", "RX8", DIR / "data" / "rcx" / "cathedral_play_buf.rcx"],
             ["RP2", "RP2", DIR / "data" / "rcx" / "button_numpad.rcx"]]
    freefield.initialize("cathedral", device=procs, zbus=False, connection="USB")
    print(freefield.SPEAKERS)
    freefield.SETUP = "cathedral"
    freefield.SPEAKERS = freefield.read_speaker_table()
    normalisation_file = DIR / "data" / "calibration" / f"calibration_cathedral_{normalisation_sound_type}_{normalisation_algorithm}.pkl"
    freefield.load_equalization(file=str(normalisation_file), frequency=False)
    normalisation_method = f"{normalisation_sound_type}_{normalisation_algorithm}"
    freefield.set_logger("DEBUG")
    if sound_type == "syllable":
        stim_DIR = DIR / "data" / "stim_files" / "tts-numbers_n13_resamp_48828"
    elif sound_type == "babble":
        stim_DIR = DIR / "data" / "stim_files" / "tts-numbers_reversed"
    elif sound_type == "pinknoise":
        stim_DIR = DIR / "data" / "stim_files" / "pinknoise"
    else:
        return

    sound_files = [file for file in stim_DIR.iterdir()]
    for file in sound_files:
        sounds.update({os.path.basename(file): slab.Sound.read(str(file))})


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

def get_speakers(n):
    if n > len(freefield.SPEAKERS):
        print(len(freefield.SPEAKERS))
        print(n)
        raise ValueError("n cannot be greater than the length of the input list")
    random_indices = np.random.choice(len(freefield.SPEAKERS), n, replace=False)
    speakers = [freefield.pick_speakers(i) for i in random_indices]
    return speakers, random_indices

def estimate_numerosity(sub_id):
    global n_sounds
    global event_id
    n_simultaneous_sounds = np.random.choice(n_sounds)
    filenames, sounds = get_sounds_with_filenames(n_simultaneous_sounds)
    speakers, speaker_indices = get_speakers(n_simultaneous_sounds)
    for i in range(n_simultaneous_sounds-1):
        freefield.apply_equalization(signal=sounds[i], speaker=speakers[i], level=True, frequency=False)
        freefield.set_signal_and_speaker(signal=sounds[i], speaker=speakers[i], equalize=False)
    freefield.play(kind=1, proc="RX81")
    """while not freefield.read(tag="response", processor="RP2"):
        time.sleep(0.05)
    response = freefield.read(tag="response", processor="RP2")"""

    response = input()

    if response == n_simultaneous_sounds:
        is_correct = True
    else:
        is_correct = False
    """save_results(event_id, sub_id, SOUND_TYPE, n_sounds, filenames, speakers, speaker_indices, is_correct)"""
    event_id += 1
    freefield.flush_buffers(processor="RX81")

def save_results(event_id, sub_id, sound_type, n_sounds, filenames, speakers, speaker_indices, is_correct):
    file_name = DIR / "data" / "results" / f"results_numerosity_judgement_{sound_type}_{sub_id}.csv"

    active_speakers = []
    for i in range(10):
        if i in speaker_indices:
            active_speakers.append(str(1))
        else:
            active_speakers.append(str(0))

    mean_speaker_distance = statistics.mean([s.distance for s in speakers])
    speaker_dist_st_dev = statistics.stdev([s.distance for s in speakers])

    active_speakers = '"' + ','.join(active_speakers) + '"'
    results = {"event_id" : event_id,
               "subject": sub_id,
               "sound_type": sound_type,
               "n_sounds" : n_sounds,
               "sound_filenames": filenames,
               "active_speakers" : active_speakers,
               "is_correct" : is_correct,
               "mean_speaker_distance": mean_speaker_distance,
               "speaker_dist_st_dev": speaker_dist_st_dev}


    df_curr_results = pd.DataFrame.from_dict(results)
    df_curr_results.to_csv(file_name, mode='a', header=not os.path.exists(file_name))
def start_trial(sub_id, n_reps):
    for i in range(n_reps):
        estimate_numerosity(sub_id)

def plot_results(sub_id, sound_type):
    data_file = DIR / "data" / "results" / f"results_numerosity_judgement_{sound_type}_{sub_id}.csv"
    try:
        with open(data_file, 'rb') as f:
            results = pd.read_csv(f)
    except Exception as e:
        print("Error:", e)
        return



