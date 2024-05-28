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
def start_trial(sub_id, n_reps=30):
    for i in range(n_reps):
        event_id = i
        filename, sound = get_sounds_with_filename(1)
        filename = filename[0]
        sound = sound[0]
        speaker = freefield.pick_speakers(np.random.randint(0, 11))[0]
        apply_mgb_equalization(signal=sound, speaker=speaker)
        freefield.play(kind=1, proc="RX81")
        response = input("Estimate Distance in m...")
        freefield.flush_buffers()
        save_results(event_id=event_id, sub_id=sub_id, response=response,
                     speaker_distance=speaker.distance, sound_filename=filename)

    return

def get_sound_with_filenames(n):
    global sounds
    if n > len(sounds):
        raise ValueError("n cannot be greater than the length of the input list")
    random_indices = np.random.choice(len(sounds), n, replace=False)
    sounds_list = list(sounds.values())
    filenames_list = list(sounds.keys())
    sounds_list = [sounds_list[i] for i in random_indices]
    filenames_list = [filenames_list[i] for i in random_indices]
    return filenames_list, sounds_list

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

def apply_mgb_equalization(signal, speaker, mgb_loudness=30, fluc=0):
    a, b, c = get_log_parameters(speaker.distance)
    signal.level = logarithmic_func(mgb_loudness + fluc, a, b, c)
    return signal

def save_results(event_id, sub_id, response, speaker_distance, sound_filename):
    file_name = DIR / "data" / "results" / f"results_localisation_accuracy_{sub_id}.csv"


    results = {"event_id": event_id,
               "sub_id": sub_id,
               "response": response,
               "speaker_distance": speaker_distance,
               "sound_filename": sound_filename}

    df_curr_results = pd.DataFrame.from_dict(results)
    df_curr_results.to_csv(file_name, mode='a', header=not os.path.exists(file_name))