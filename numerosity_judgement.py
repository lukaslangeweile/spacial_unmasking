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
import pandas as pd
import matplotlib.pyplot as plt
import glob
from pathlib import Path


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
n_sounds = [2, 3, 4, 5, 6]
SOUND_TYPE = None
sounds = {}
maximum_n_samples = 0

def initialize_setup(normalisation_algorithm="rms", normalisation_sound_type="syllable", sound_type="countries"):
    global normalisation_method
    global sounds
    global SOUND_TYPE
    global maximum_n_samples
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
    elif sound_type == "countries":
        stim_DIR = DIR / "data" / "stim_files" / "tts-countries_n13_resamp_48828"
    else:
        return

    sound_files = [file for file in stim_DIR.iterdir()]
    for file in sound_files:
        sounds.update({os.path.basename(file): slab.Sound.read(str(file))})
    maximum_n_samples = max([slab.Sound.read(str(file)).n_samples for file in sound_files])

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
    speakers = [freefield.pick_speakers(i)[0] for i in random_indices]
    return speakers, random_indices

def estimate_numerosity(sub_id):
    global n_sounds
    global event_id
    fluctuation = np.random.uniform(-1, 1)
    n_simultaneous_sounds = np.random.choice(n_sounds)
    filenames, sounds = get_sounds_with_filenames(n_simultaneous_sounds)
    speakers, speaker_indices = get_speakers(n_simultaneous_sounds)
    for i in range(n_simultaneous_sounds):
        apply_mgb_equalization(signal=sounds[i], speaker=speakers[i], fluc=fluctuation)
        freefield.set_signal_and_speaker(signal=sounds[i], speaker=speakers[i], equalize=False)
        print(sounds[i].n_samples)
        time.sleep(0.1)
    time.sleep(0.3)
    freefield.play(kind=1, proc="RX81")
    print(f"simulatneous_sounds = {n_simultaneous_sounds}")
    while not freefield.read(tag="response", processor="RP2"):
        time.sleep(0.05)
    response = freefield.read(tag="response", processor="RP2")
    freefield.flush_buffers(processor="RX81", maximum_n_samples=maximum_n_samples)
    freefield.write(tag="playbuflen", value=30000, processors="RX81")


    if response == n_simultaneous_sounds:
        is_correct = True
    else:
        is_correct = False
    save_results(event_id, sub_id, SOUND_TYPE, n_simultaneous_sounds, response, is_correct, speakers)
    event_id += 1
    print(f"simulatneous_sounds = {n_simultaneous_sounds}")
    time.sleep(1.0)


def save_results(event_id, sub_id, sound_type, n_sounds, response, is_correct, speakers):
    file_name = DIR / "data" / "results" / f"results_numerosity_judgement_{sound_type}_{sub_id}.csv"

    """active_speakers = []
    for i in range(10):
        if i in speaker_indices:
            active_speakers.append(str(1))
        else:
            active_speakers.append(str(0))"""

    mean_speaker_distance = statistics.mean([s.distance for s in speakers])
    if n_sounds == 1:
        speaker_dist_st_dev = 0
    else:
        speaker_dist_st_dev = statistics.stdev([s.distance for s in speakers])

    """active_speakers = '"' + ','.join(active_speakers) + '"'"""
    results = {"event_id": event_id,
               "subject": sub_id,
               "sound_type": sound_type,
               "n_sounds": n_sounds,
               "response": response,
               "is_correct": is_correct,
               "mean_speaker_distance": mean_speaker_distance,
               "speaker_dist_st_dev": speaker_dist_st_dev}


    df_curr_results = pd.DataFrame(results, index=[0])
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

def apply_mgb_equalization(signal, speaker, mgb_loudness=25, fluc=0):
    a, b, c = get_log_parameters(speaker.distance)
    signal.level = logarithmic_func(mgb_loudness + fluc, a, b, c)
    return signal


def plot_averaged_responses(sub_ids, sound_type):
    # Initialize an empty DataFrame to hold all data
    all_data = pd.DataFrame()

    # Loop through each subject ID and load the corresponding data
    for sub_id in sub_ids:
        file_pattern = DIR / "data" / "results" / f"results_numerosity_judgement_{sound_type}_{sub_id}.csv"
        files = glob.glob(str(file_pattern))

        df_list = [pd.read_csv(file) for file in files if pd.read_csv(file).shape[0] > 0]  # Ensure non-empty files
        if df_list:
            df = pd.concat(df_list, ignore_index=True)
            all_data = pd.concat([all_data, df], ignore_index=True)

    if all_data.empty:
        print("No valid data found for the given subject IDs.")
        return

    # Calculate the mean response for each n_sounds
    mean_responses = all_data.groupby('n_sounds')['response'].mean().reset_index()

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(mean_responses['n_sounds'], mean_responses['response'], marker='o')
    plt.xlabel('Number of Sounds (n_sounds)')
    plt.ylabel('Average Response')
    plt.title(f'Average Responses vs. Number of Sounds for {sound_type}')
    plt.grid(True)
    plt.show()

"""plot_averaged_responses([100, 9], "countries")
plot_averaged_responses(100, "countries")
plot_averaged_responses(9, "countries")"""