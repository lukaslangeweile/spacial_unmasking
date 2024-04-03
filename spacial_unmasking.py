import freefield
import slab
import os
import pathlib
import time
import numpy
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

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


def initialize_setup(normalisation_algorithm="rms", normalisation_sound_type="syllable"):
    global normalisation_method
    procs = [["RX81", "RX8", DIR / "data" / "rcx" / "cathedral_play_buf.rcx"],
             ["RP2", "RP2", DIR / "data" / "rcx" / "button_numpad.rcx"]]
    freefield.initialize("cathedral", device=procs, zbus=False, connection="USB")
    normalisation_file = DIR / "data" / "calibration" / f"calibration_cathedral_{normalisation_sound_type}_{normalisation_algorithm}.pkl"
    freefield.load_equalization(file=str(normalisation_file), frequency=False)
    normalisation_method = f"{normalisation_sound_type}_{normalisation_algorithm}"
    freefield.set_logger("DEBUG")

def start_trial(sub_id, masker_type, stim_type):
    global normalisation_method
    target_speaker = freefield.pick_speakers(5)[0]
    talker = numpy.random.choice(talkers)
    spacial_unmask_from_peripheral_speaker(start_speaker=0, target_speaker=target_speaker, sub_id=sub_id,
                                           masker_type=masker_type, stim_type=stim_type, talker=talker,
                                           normalisation_method=normalisation_method)
    spacial_unmask_from_peripheral_speaker(start_speaker=10, target_speaker=target_speaker, sub_id=sub_id,
                                          masker_type=masker_type, stim_type=stim_type, talker=talker,
                                          normalisation_method=normalisation_method)

def get_correct_response(file):
    for key in num_dict:
        if key in str(file):
            return num_dict.get(key)

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
        babble_DIR = DIR / "data" / "stim_files" / "tts-numbers_reversed"
        contents = [file for file in babble_DIR.iterdir()]
        masker_file = os.path.join(DIR, get_random_file(contents))
    else:
        masker_file = None
    return masker_file

def get_random_file(files):
    return numpy.random.choice(files)

def get_target_and_masker_file(sex=None, number=None, talker=None):
    stimuli = get_possible_files(sex=sex, number=number, talker=talker)
    target_file = get_random_file(stimuli)
    correct_response = get_correct_response(target_file)
    masker_file = get_random_file(get_possible_files(number=correct_response, exclude=True))
    return target_file, masker_file

def spacial_unmask_from_peripheral_speaker(start_speaker, target_speaker, sub_id, masker_type, stim_type, talker, normalisation_method):
    global event_id
    if start_speaker > 5:
        iterator = list(range(10, 5, -1))
    else:
        iterator = list(range(5))

    numpy.random.shuffle(iterator)
    talker = talker

    for i in iterator:
        masking_speaker = freefield.pick_speakers(i)[0]
        stairs = slab.Staircase(start_val=3, n_reversals=5, step_sizes=[7, 5, 3, 1])

        for level in stairs:
            if masker_type != "syllable":
                masker_file = get_non_syllable_masker_file(masker_type)
                target_file = get_target_and_masker_file(talker=talker)[0]
            else:
                target_file, masker_file = get_target_and_masker_file(talker=talker)
            masker = slab.Sound.read(masker_file)
            print(masker.samplerate)
            target = slab.Sound.read(target_file)
            print(target.samplerate)
            freefield.apply_equalization(signal=masker, speaker=masking_speaker, level=True, frequency=False)
            freefield.apply_equalization(signal=target, speaker=target_speaker, level=True, frequency=False)
            target.level += level  # TODO: think about which level needs to be adjusted
            freefield.set_signal_and_speaker(signal=target, speaker=target_speaker, equalize=False)
            freefield.set_signal_and_speaker(signal=masker, speaker=masking_speaker, equalize=False)
            freefield.play(kind=1, proc="RX81")
            while not freefield.read("response", "RP2"):
                time.sleep(0.05)
            response = freefield.read("response", "RP2")

            if response == get_correct_response(target_file):
                stairs.add_response(1)
            else:
                stairs.add_response(0)

            freefield.flush_buffers(processor="RX81")
            time.sleep(2.5)

        save_results(event_id=event_id ,sub_id=sub_id, threshold=stairs.threshold(), distance_masker=masking_speaker.distance,
                     distance_target=target_speaker.distance, level_masker=masker.level, level_target=target.level,
                     masker_type=masker_type, stim_type=stim_type, talker=talker, normalisation_method=normalisation_method,
                     normalisation_level_masker=masking_speaker.level, normalisation_level_target=target_speaker.level)
        print(event_id)
        event_id += 1

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
        level_masker = numpy.mean(level_masker)
    if len(level_target.shape) == 1:
        level_target = numpy.mean(level_target)

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
    df_curr_results.to_csv(file_name, mode='a', header=not os.path.exists(file_name), index=False)
    """
    try:
        # Ensure the directory structure exists
        os.makedirs(os.path.dirname(file_name), exist_ok=True)

        with open(file_name, 'ab') as f:  # Append mode
            pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
            print("Data appended to pickle file successfully.")
    except Exception as e:
        print("Error:", e)
    """

def plot_target_ratio_vs_distance(sub_id, masker_type):
    data_file = DIR / "data" / "results" / f"results_spacial_unmasking_{sub_id}.csv"
    try:
        with open(data_file, 'rb') as f:
            results = pd.read_csv(f)
    except Exception as e:
        print("Error:", e)
        return

    results["target_to_tasker_ratio"] = results["level_target"] / results["level_masker"]
    results["target_normalisation_adapted_ratio"] = (results["level_target"] / results["normalisation_level_target"])
    # Plot
    sns.scatterplot(data=results, x="distance_masker", y="target_normalisation_adapted_ratio")
    plt.xlabel("Distance of Masking Speaker")
    plt.ylabel("Ratio of Target Level")
    plt.title("Ratio of Target Level vs Distance of Masking Speaker")
    fig = plt.gcf()
    plt.show()
    plt.draw()
    fig.savefig(DIR / "data" / "results" / "figs" /f"results_{sub_id}.pdf")

