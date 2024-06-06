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

import util

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
talkers = ["p245", "p248", "p256", "p268", "p284", "p307", "p318"]


def start_trial(sub_id, masker_type, stim_type):
    target_speaker = freefield.pick_speakers(5)[0]
    talker = np.random.choice(talkers)
    train_talker(talker)
    input("Start Experiment by pressing Enter...")
    spacial_unmask_within_range(nearest_speaker=0, farthest_speaker=0, target_speaker=target_speaker, sub_id=sub_id,
                                           masker_type=masker_type, stim_type=stim_type, talker=talker,
                                           normalisation_method="mgb_normalisation")

def get_possible_files(sex=None, number=None, talker=None, exclude=False):
    possible_files = []
    stim_dir = DIR / "data" / "stim_files" / "tts-numbers_n13_resamp_48828"  #TODO: Add files and directory
    if isinstance(number, int):
        for key, val in num_dict.items():
            if val == number:
                number = key

    if not exclude:
        for file in os.listdir(stim_dir):
            if sex is not None and sex not in str(file):
                continue
            if number is not None and number not in str(file):
                continue
            if talker is not None and talker not in str(file):
                continue
            possible_files.append(os.path.join(stim_dir, file))

    else:
        for file in os.listdir(stim_dir):
            if sex is not None and sex in str(file):
                continue
            if number is not None and number in str(file):
                continue
            if talker is not None and talker in str(file):
                continue
            possible_files.append(os.path.join(stim_dir, file))

    return possible_files

def get_non_syllable_masker_file(masker_type):
    if masker_type == "pinknoise":
        pink_noise_DIR = DIR / "data" / "stim_files" / "pinknoise" #placeholder
        contents = [file for file in pink_noise_DIR.iterdir()]
        masker_file = os.path.join(DIR, get_random_file(contents))
    elif masker_type == "babble":
        babble_DIR = DIR / "data" / "stim_files" / "babble-numbers-reversed-n13-shifted_resamp_48828"
        contents = [file for file in babble_DIR.iterdir()]
        masker_file = os.path.join(DIR, get_random_file(contents))
    else:
        masker_file = None
    return masker_file

def get_random_file(files):
    return np.random.choice(files)

def get_target_number_file(sex=None, number=None, talker=None):
    stimuli = get_possible_files(sex=sex, number=number, talker=talker)
    target_file = get_random_file(stimuli)
    return target_file

def spacial_unmask_within_range(nearest_speaker, farthest_speaker, target_speaker, sub_id, masker_type, stim_type, talker, normalisation_method):
    global event_id
    print("Beginning spacial unmasking")
    iterator = list(range(nearest_speaker, farthest_speaker+1))
    np.random.shuffle(iterator)
    talker = talker

    for i in iterator:

        masking_speaker = freefield.pick_speakers(i)[0]
        stairs = slab.Staircase(start_val=-5, n_reversals=16, step_sizes=[7, 5, 3, 1])

        for level in stairs:
            masker_file = get_non_syllable_masker_file(masker_type)
            target_file = get_target_number_file(talker=talker)
            masker = slab.Sound.read(masker_file)
            masker = slab.Sound(masker.data[:, 0])
            target = slab.Sound.read(target_file)
            print(masker.samplerate)
            print(target.samplerate)
            masker = util.apply_mgb_equalization(signal=masker, speaker=masking_speaker)
            target = util.apply_mgb_equalization(signal=target, speaker=target_speaker)
            target.level += level  # TODO: think about which level needs to be adjusted

            if masking_speaker == target_speaker:

                max_length = max(len(masker.data), len(target.data))

                # Pad both arrays with zeros to make them the same length
                masker_padded = np.pad(masker.data, ((0, max_length - len(masker.data)), (0, 0)), 'constant')
                target_padded = np.pad(target.data, ((0, max_length - len(target.data)), (0 , 0)), 'constant')
                to_play_data = np.array(masker_padded) + np.array(target_padded)
                to_play = slab.Sound(data=to_play_data)
                """
                to_play = freefield.apply_equalization(signal=to_play, speaker=target_speaker, frequency=False)
                to_play.level = to_play.level + 3
                """
                freefield.set_signal_and_speaker(signal=to_play, speaker=target_speaker, equalize=False)
            else:
                freefield.set_signal_and_speaker(signal=target, speaker=target_speaker, equalize=False)
                freefield.set_signal_and_speaker(signal=masker, speaker=masking_speaker, equalize=False)
            freefield.play(kind=1, proc="RX81")
            while not freefield.read("response", "RP2"):
                time.sleep(0.05)
            response = freefield.read("response", "RP2")

            if response == util.get_correct_response_number(target_file):
                stairs.add_response(1)
            else:
                stairs.add_response(0)

            save_per_response(event_id=event_id, sub_id=sub_id, step_number=stairs.this_trial_n, level_masker=masker.level,
                              level_target=target.level, distance_masker=masking_speaker.distance,
                              distance_target=target_speaker.distance, masker_filename=masker_file, target_filename=target_file,
                              normalisation_method=normalisation_method, normalisation_level_masker=get_speaker_normalisation_level(masking_speaker),
                              normalisation_level_target=get_speaker_normalisation_level(target_speaker),
                              played_number=util.get_correct_response_number(target_file),
                              response_number=response)

            """freefield.flush_buffers(processor="RX81")"""
            freefield.write(tag="data0", value=0, processors="RX81")
            freefield.write(tag="data1", value=0, processors="RX81")
            time.sleep(2.5)

        save_results(event_id=event_id ,sub_id=sub_id, threshold=stairs.threshold(n=10), distance_masker=masking_speaker.distance,
                     distance_target=target_speaker.distance, level_masker=masker.level,
                     level_target=target_speaker.level + stairs.threshold(n=10),
                     masker_type=masker_type, stim_type=stim_type, talker=talker, normalisation_method=normalisation_method,
                     normalisation_level_masker=get_speaker_normalisation_level(masking_speaker),
                     normalisation_level_target=get_speaker_normalisation_level(target_speaker))
        print(event_id)
        event_id += 1

def get_key_by_value(value):
    return next((key for key, val in num_dict.items() if val == value), None)

def train_talker(talker_id):
    numbers = ["one", "two", "three", "four", "five", "six", "eight", "nine"]
    talker_num_dict = {}
    for number in numbers:
        filepath = os.path.join(DIR, get_possible_files(number=number, talker=talker_id)[0])
        talker_num_dict.update({number: filepath})
    print(talker_num_dict)
    for i in range(11):
        while not freefield.read("response", "RP2"):
            time.sleep(0.05)
        button_press = freefield.read("response", "RP2")
        button_press_written = get_key_by_value(button_press)
        print(button_press_written)
        print(talker_num_dict.get(button_press_written))
        signal = slab.Sound.read(talker_num_dict.get(button_press_written))
        signal = apply_mgb_equalization(signal=signal, speaker=freefield.pick_speakers(5)[0])
        freefield.set_signal_and_speaker(signal=signal, speaker=i, equalize=False)
        freefield.play(kind=1, proc="RX81")
        time.sleep(0.5)
        """freefield.flush_buffers(processor="RX81")"""
        freefield.write(tag="data0", value=0, processors="RX81")
        freefield.write(tag="data1", value=0, processors="RX81")
    time.sleep(1.0)

def set_event_id(new_event_id):
    global event_id
    event_id = new_event_id

def save_results(event_id, sub_id, threshold, distance_masker, distance_target,
                 level_masker, level_target, masker_type, stim_type, talker,
                 normalisation_method, normalisation_level_masker,
                 normalisation_level_target,):
    file_name = DIR / "data" / "results" / f"results_spacial_unmasking_{sub_id}.csv"

    # Check if the file exists
    if file_name.exists():
        # Load existing data from CSV file into a DataFrame
        df_curr_results = pd.read_csv(file_name)
    else:
        # If the file doesn't exist, create an empty DataFrame
        df_curr_results = pd.DataFrame()

    if len(level_masker.shape) == 1:
        level_masker = np.mean(level_masker)
    if len(level_target.shape) == 1:
        level_target = np.mean(level_target)

    # Convert necessary columns to desired data types
    event_id = int(event_id)
    threshold = float(threshold)
    sub_id = int(sub_id)
    level_masker = float(level_masker)
    level_target = float(level_target)
    distance_masker = float(distance_masker)
    distance_target = float(distance_target)
    normalisation_level_target = float(normalisation_level_target)
    normalisation_level_masker = float(normalisation_level_masker)
    masker_type = str(masker_type)
    stim_type = str(stim_type)
    talker = str(talker)
    normalisation_method = str(normalisation_method)

    new_row = {"event_id" : event_id,
            "subject": sub_id,
            "threshold": threshold,
            "distance_masker": distance_masker,
            "distance_target": distance_target,
            "level_masker": level_masker, "level_target": level_target,
            "masker_type": masker_type, "stim_type": stim_type,
            "talker": talker,
            "normalisation_method": normalisation_method,
            "normalisation_level_masker": normalisation_level_masker,
            "normalisation_level_target": normalisation_level_target}

    df_curr_results = df_curr_results.append(new_row, ignore_index=True)
    df_curr_results.to_csv(file_name, mode='w', header=True, index=False)

def save_per_response(event_id, sub_id, step_number, level_masker, level_target,
                      distance_masker, distance_target, masker_filename, target_filename,
                      normalisation_method, normalisation_level_masker, normalisation_level_target,
                      played_number, response_number):

    is_correct = played_number == response_number

    file_name = DIR / "data" / "results" / f"results_per_step_spacial_unmasking_{sub_id}.csv"

    # Check if the file exists
    if file_name.exists():
        # Load existing data from CSV file into a DataFrame
        df_curr_results = pd.read_csv(file_name)
    else:
        # If the file doesn't exist, create an empty DataFrame
        df_curr_results = pd.DataFrame()

    if len(level_masker.shape) == 1:
        level_masker = np.mean(level_masker)
    if len(level_target.shape) == 1:
        level_target = np.mean(level_target)

    new_row = {"event_id": event_id,
               "subject": sub_id,
               "step_number:": step_number,
               "distance_masker": distance_masker,
               "distance_target": distance_target,
               "level_masker": level_masker, "level_target": level_target,
               "masker_filename": masker_filename, "target_filename": target_filename,
               "normalisation_method": normalisation_method,
               "normalisation_level_masker": normalisation_level_masker,
               "normalisation_level_target": normalisation_level_target,
               "played_number": played_number, "response_number": response_number,
               "is_correct": is_correct}

    df_curr_results = df_curr_results.append(new_row, ignore_index=True)
    df_curr_results.to_csv(file_name, mode='w', header=True, index=False)

def plot_target_ratio_vs_distance(sub_id, masker_type):
    data_file = DIR / "data" / "results" / f"results_spacial_unmasking_{sub_id}.csv"
    try:
        with open(data_file, 'rb') as f:
            results = pd.read_csv(f)
    except Exception as e:
        print("Error:", e)
        return

    results["target_to_masker_ratio"] = results["level_target"] - results["normalisation_level_target"]
    results["target_normalisation_adapted_ratio"] = results["threshold"] / results["normalisation_level_target"]

    results_first_run = results[results["event_id"] < 10]
    results_second_run = results[results["event_id"] >= 10]
    # Plot

    sns.scatterplot(data=results_first_run, x="distance_masker", y="target_to_masker_ratio", palette="blue")
    sns.scatterplot(data=results_second_run, x="distance_masker", y="target_to_masker_ratio", color="red")

    plt.xlabel("Distance of Masking Speaker")
    plt.ylabel("Ratio of Target Level")
    plt.title("Ratio of Target Level vs Distance of Masking Speaker")
    fig = plt.gcf()
    plt.show()
    plt.draw()
    fig.savefig(DIR / "data" / "results" / "figs" /f"results_{sub_id}.pdf")

def plot_average_results(sub_ids="all"):
    result_filepath = DIR / "data" / "results"
    data_files = []
    results = pandas.DataFrame()
    speaker_distances = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    masker_types = ["babble", "pinknoise"]

    if sub_ids == "all":
        data_files = [file for file in result_filepath.iterdir()]
    elif isinstance(sub_ids, list):
        for sub_id in sub_ids:
            filepath = result_filepath / f"results_spacial_unmasking_{sub_id}.csv"
            data_files.append(filepath)
    else:
        print("Plotting average is only possible with all subject ids or with a list of selected subject ids!")

    for file in data_files:
        try:
            with open(file, 'rb') as f:
                result = pd.read_csv(f)
        except Exception as e:
            print("Error:", e)
            return
        results = pd.concat([results, result], ignore_index=True)

    results["target_normalisation_adapted_ratio"] = (results["level_target"] - results["normalisation_level_target"])

    grouped = pd.DataFrame(results).groupby("masker_type")
    grouped_babble = grouped.get_group("babble").groupby("distance_masker")
    """grouped_pinknoise = grouped.get_group("pinknoise").groupby("distance_masker")"""
    average_tnr_babble = grouped_babble["target_normalisation_adapted_ratio"].mean()
    """average_tnr_pinknoise = grouped_pinknoise["target_normalisation_adapted_ratio"].mean()"""

    sns.scatterplot(x=average_tnr_babble.index, y=average_tnr_babble.values, color='blue', alpha=0.5, label='Group A')
    """sns.scatterplot(x=average_tnr_pinknoise.index, y=average_tnr_pinknoise.values, color='red', alpha=0.5, label='Group B')"""

    plt.xlabel("Distance of Masking Speaker")
    plt.ylabel("Ratio of Target Level")
    plt.title("Average Ratio of Target Level vs Distance of Masking Speaker")
    fig = plt.gcf()
    plt.show()
    plt.draw()
    fig.savefig(DIR / "data" / "results" / "figs" / f"average_results_.pdf")

def get_speaker_normalisation_level(speaker, mgb_loudness=30):
    a, b, c = get_log_parameters(speaker.distance)
    return logarithmic_func(x=mgb_loudness, a=a, b=b, c=c)