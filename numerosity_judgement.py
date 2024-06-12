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
import util


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

n_sounds = [2, 3, 4, 5, 6]
maximum_n_samples = 0

def estimate_numerosity(sub_id, block_id, trial_index, stim_type="countries_forward"):
    global n_sounds
    global event_id
    sounds = util.get_sounds_dict(stim_type)
    fluctuation = np.random.uniform(-1, 1)
    n_simultaneous_sounds = np.random.choice(n_sounds)
    filenames, sounds = util.get_sounds_with_filenames(sounds_dict=sounds, n=n_simultaneous_sounds, randomize=True)
    speakers, speaker_indices = util.get_n_random_speakers(n_simultaneous_sounds)
    util.set_multiple_signals(signals=sounds, speakers=speakers, equalize=True, fluc=fluctuation)
    freefield.play(kind=1, proc="RX81")
    util.start_timer()
    print(f"simulatneous_sounds = {n_simultaneous_sounds}")
    while not freefield.read(tag="response", processor="RP2"):
        time.sleep(0.05)
    response = freefield.read(tag="response", processor="RP2")
    reaction_time = util.get_elapsed_time()
    if response == n_simultaneous_sounds:
        is_correct = True
    else:
        is_correct = False
    save_results(event_id=event_id, sub_id=sub_id, trial_index=trial_index, block_id=block_id, stim_type=stim_type, filenames=filenames,
                 speaker_ids=speaker_indices, n_sounds=n_simultaneous_sounds, response=response, is_correct=is_correct, speakers=speakers, reaction_time=reaction_time)
    event_id += 1
    print(f"simulatneous_sounds = {n_simultaneous_sounds}")


def save_results(event_id, sub_id, trial_index, block_id, stim_type, filenames, speaker_ids, n_sounds, response, is_correct, speakers, reaction_time):
    file_name = DIR / "data" / "results" / f"results_numerosity_judgement_{stim_type}_{sub_id}.csv"

    mean_speaker_distance = statistics.mean([s.distance for s in speakers])
    if n_sounds == 1:
        speaker_dist_st_dev = 0
    else:
        speaker_dist_st_dev = statistics.stdev([s.distance for s in speakers])

    stim_levels = []
    for id in speaker_ids:
        stim_levels.append(util.get_speaker_normalisation_level(freefield.pick_speakers(id)[0]))

    stim_country_ids = []
    stim_talker_ids = []
    for filename in filenames:
        talker, sex, text = util.parse_country_or_number_filename(filename)
        stim_country_ids.append(text)
        stim_talker_ids.append(talker)

    results = {"event_id": None,
               "subject_id": sub_id,
               "timestamp": util.get_timestamp(),
               "session_index": 3,
               "plane": "distance",
               "setup": "cathedral",
               "task": "numerosity_judgement",
               "block": block_id,
               "trial_index": trial_index,
               "stim_number": n_sounds,
               "stim_type": stim_type,
               "stim_country_ids": str(stim_country_ids),
               "stim_talker_ids": str(stim_talker_ids),
               "speaker_ids": str(speaker_ids),
               "stim_level": str(stim_levels), #TODO: ask about this
               "resp_number": response,
               "is_correct": is_correct,
               "speaker_distance_mean": mean_speaker_distance,
               "speaker_distance_st_dev": speaker_dist_st_dev,
               "reaction_time": reaction_time}

    # Debug: Print key, value, and type of each entry in the results dictionary
    for key, value in results.items():
        print(f"Key: {key}, Value: {value}, Type: {type(value)}")

    df_curr_results = pd.DataFrame(results, index=[0])
    df_curr_results.to_csv(file_name, mode='a', header=not os.path.exists(file_name))

def start_experiment(sub_id, block_id, stim_type, n_reps):
    trial_index = 0
    for i in range(n_reps):
        estimate_numerosity(sub_id, block_id, trial_index, stim_type)
        trial_index += 1

def plot_results(sub_id, sound_type):
    data_file = DIR / "data" / "results" / f"results_numerosity_judgement_{sound_type}_{sub_id}.csv"
    try:
        with open(data_file, 'rb') as f:
            results = pd.read_csv(f)
    except Exception as e:
        print("Error:", e)
        return


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

if __name__ == "__main__":
    plot_averaged_responses([100, 9], "countries_forward")
