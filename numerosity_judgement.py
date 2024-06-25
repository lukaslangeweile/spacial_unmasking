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
import logging
from scipy.stats import gaussian_kde
import ast
import re


normalisation_method = None
event_id = 0
DIR = pathlib.Path(os.curdir)

condition_counter = {2: 0,
                     3: 0,
                     4: 0,
                     5: 0,
                     6: 0}

logging.basicConfig(filename="numerosity_judgement.log", level=logging.ERROR)

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


def estimate_numerosity(sub_id, block_id, stim_type, n_reps):
    valid_responses = [2, 3, 4, 5, 6]
    df = pd.read_csv(DIR / "data" / "spectral_coverage_data" / "tts_spectral_coverage.csv")
    condition_dict = create_condition_bin_dict(df=df, stim_type=stim_type, reps=n_reps)

    global event_id
    seq = slab.Trialsequence(conditions=[2, 3, 4, 5, 6], n_reps=n_reps)
    for trial in seq:
        trial_index = seq.this_n
        stim_dir = util.get_stim_dir(stim_type=stim_type)
        sounds_dict = util.get_sounds_dict(stim_type=stim_type)
        fluctuation = np.random.uniform(-1, 1)
        n_simultaneous_sounds = trial # TODO: is this correct?
        max_n_samples = util.get_max_n_samples(stim_dir)
        """filenames, sounds = util.get_sounds_with_filenames(sounds_dict=sounds, n=n_simultaneous_sounds, randomize=True)
        speakers, speaker_indices = util.get_n_random_speakers(n_simultaneous_sounds)"""
        sounds, filenames, speaker_indices, spectral_coverage = get_pseudo_randomized_stimuli(trial, condition_dict, n_reps, stim_type)

        speakers = freefield.pick_speakers(speaker_indices)
        logging.info(f"Presenting {n_simultaneous_sounds} sounds at speakers with indices {speaker_indices}. "
                     f"Trial index = {trial_index}")

        util.set_multiple_signals(signals=sounds, speakers=speakers, equalize=True, fluc=fluctuation, max_n_samples=max_n_samples)
        freefield.play(kind=1, proc="RX81")
        util.start_timer()
        """time.sleep(max_n_samples / 24414.0)"""
        """response = input("Enter number between 2 and 6")"""
        response = None
        while True:
            response = freefield.read("response", "RP2")
            time.sleep(0.05)
            if response in valid_responses:
                break
        reaction_time = util.get_elapsed_time()
        if int(response) == int(n_simultaneous_sounds):
            is_correct = True
        else:
            is_correct = False

        logging.info(f"Got response = {response}. Number of played sounds = {n_simultaneous_sounds}. is_correct = {is_correct}")
        save_results(event_id=event_id, sub_id=sub_id, trial_index=trial_index, block_id=block_id, stim_type=stim_type, filenames=filenames,
                     speaker_ids=speaker_indices, n_sounds=n_simultaneous_sounds, response=response, is_correct=is_correct, speakers=speakers,
                     reaction_time=reaction_time, spectral_coverage=spectral_coverage)
        event_id += 1
        print(f"simulatneous_sounds = {n_simultaneous_sounds}")


def save_results(event_id, sub_id, trial_index, block_id, stim_type, filenames, speaker_ids, n_sounds, response, is_correct, speakers, reaction_time, spectral_coverage):
    try:
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
        stim_sexes = []
        for filename in filenames:
            talker, sex, text = util.parse_country_or_number_filename(filename)
            stim_country_ids.append(text)
            stim_talker_ids.append(talker)
            stim_sexes.append(sex)

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
                   "stim_sexes": str(stim_sexes),
                   "stim_talker_ids": str(stim_talker_ids),
                   "speaker_ids": str(speaker_ids),
                   "stim_level": str(stim_levels), #TODO: ask about this
                   "resp_number": response,
                   "is_correct": is_correct,
                   "speaker_distance_mean": mean_speaker_distance,
                   "speaker_distance_st_dev": speaker_dist_st_dev,
                   "reaction_time": reaction_time,
                   "spectral_coverage": spectral_coverage}


        df_curr_results = pd.DataFrame(results, index=[0])
        df_curr_results.to_csv(file_name, mode='a', header=not os.path.exists(file_name))
    except FileNotFoundError as e:
        logging.error(f"Result file not found: {e}")
        print(f"Result file not found: {e}")
    except Exception as e:
        logging.error(f"An error occurred in save_results: {e}")
        print(f"An error occurred: {e}")

def start_experiment(sub_id, block_id, stim_type, n_reps=10):
    logging.info("Starting numerosity judgement experiment.")
    estimate_numerosity(sub_id, block_id, stim_type, n_reps)


def plot_results(sub_id, sound_type):
    data_file = DIR / "data" / "results" / f"results_numerosity_judgement_{sound_type}_{sub_id}.csv"
    try:
        with open(data_file, 'rb') as f:
            results = pd.read_csv(f)
    except Exception as e:
        print("Error:", e)
        return


def plot_averaged_responses(sub_ids, sound_type):
    try:
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
    except FileNotFoundError as e:
        logging.error(f"One or more result file(s) not found: {e}")
        print(f"One or more result file(s) not found: {e}")
    except Exception as e:
        logging.error(f"An error occurred in plot_averaged_responses: {e}")
        print(f"An error occurred: {e}")

def count_condition_rep(condition):
    global condition_counter
    for con in condition_counter.keys():
        if condition == con:
            current_count = condition_counter.get(con)
            new_value = current_count + 1
            condition_counter.update({con: new_value})
            return current_count

def create_condition_bin_dict(df, stim_type, reps):
    condition_dict = {}
    df_filtered = df[(df["setup"] == "cathedral") & (df["stim_type"] == stim_type)]
    for i in range(2, 7):
        con_df = df_filtered[(df_filtered["n_presented"] == i)]
        con_df = con_df.sort_values(by="spectral_coverage", ascending=True)
        start_value = con_df.iloc[1]["spectral_coverage"]
        end_value = con_df.iloc[-1]["spectral_coverage"]
        con_bin_list = list()
        bin_size = (end_value - start_value) / float(reps)
        for j in range(reps):
            lower_bound = start_value + j * bin_size
            upper_bound = start_value + (j + 1) * bin_size

            bin_df = con_df[(con_df["spectral_coverage"] >= lower_bound) & (con_df["spectral_coverage"] <= upper_bound)]
            con_bin_list.append(bin_df)
            print(f"bin_length = {len(bin_df)}")
        np.random.shuffle(con_bin_list)
        key = i
        condition_dict.update({key: con_bin_list})
    return condition_dict

def get_pseudo_randomized_stimuli(condition, condition_dict, n_reps, stim_type):
    current_count = count_condition_rep(condition)
    con_bin_list = condition_dict.get(condition)
    bin = con_bin_list[current_count]
    stim_dir = util.get_stim_dir(stim_type=stim_type)
    if len(bin) > 0:
        random_row = bin.sample().iloc[0]
        recording_filenames = ast.literal_eval(random_row["file_names"])
        filenames_list = [parse_recording_filename(rec, only_filename=True) for rec in recording_filenames]
        sounds_list = [slab.Sound.read(os.path.join(stim_dir, filename)) for filename in filenames_list]
        speaker_distances = ast.literal_eval(random_row["distances"])
        speaker_ids = [int(distance-2) for distance in speaker_distances]
        spectral_coverage = random_row["spectral_coverage"]
        logging.info(f"Got pseudo-randomized stimuli for condition n_sounds = {condition}.")
        logging.info(f"Length of rows in bin = {len(bin)}.")
        logging.info(f"Spectral coverage = {spectral_coverage}.")
    else:
        sounds_dict = util.get_sounds_dict(stim_type=stim_type)
        sounds_list, filenames_list = util.get_sounds_with_filenames(sounds_dict=sounds_dict, n=condition, randomize=True)
        speakers, speaker_ids = util.get_n_random_speakers(n=condition)
        spectral_coverage = util.get_spectral_coverage(filenames_list, speaker_ids, stim_type)
    return sounds_list, filenames_list, speaker_ids, spectral_coverage

def parse_recording_filename(rec_filename, only_filename=True): #TODO: revisit pattern and filename if not working
    # Define the regex pattern
    pattern = r"^sound-(.*?)_mgb-level-(\d+)_distance-([\d.]+)\.wav$"

    # Match the pattern with the filename
    match = re.match(pattern, rec_filename)

    if match:
        filename = match.group(1) + ".wav"
        mgb_level = int(match.group(2))
        distance = float(match.group(3))
        if only_filename:
            return filename
        else:
            return filename, mgb_level, distance
    else:
        raise ValueError("Filename does not match the expected pattern")

if __name__ == "__main__":
    plot_averaged_responses([100, 9], "countries_forward")
