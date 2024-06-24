import freefield
import pandas as pd
import slab
import os
import pathlib
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import util
import logging

# Configure logging
logging.basicConfig(filename='spatial_unmasking.log', level=logging.ERROR)

event_id = 0
DIR = pathlib.Path(os.curdir)
num_dict = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 7, "eight": 8, "nine": 9}
target_dict = {}
talkers = ["p245", "p248", "p268", "p284"]


def start_experiment(sub_id, masker_type, stim_type):
    try:
        target_speaker = freefield.pick_speakers(5)[0]
        talker = np.random.choice(talkers)
        train_talker(talker)
        input("Start Experiment by pressing Enter...")

        spacial_unmask_within_range(speaker_indices=[0, 3, 5, 7, 10], target_speaker=target_speaker,
                                    sub_id=sub_id, masker_type=masker_type, stim_type=stim_type,
                                    talker=talker,
                                    normalisation_method="mgb_normalisation")
    except Exception as e:
        logging.error(f"An error occurred in start_experiment: {e}")
        print(f"An error occurred: {e}")


def get_correct_response(file):
    try:
        for key in num_dict:
            if key in str(file):
                return num_dict.get(key)
    except Exception as e:
        logging.error(f"An error occurred in get_correct_response: {e}")
        print(f"An error occurred: {e}")


def get_possible_files(sex=None, number=None, talker=None, exclude=False):
    possible_files = []
    try:
        stim_dir = DIR / "data" / "stim_files" / "tts-numbers_n13_resamp_24414"  # TODO: Add files and directory
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
    except Exception as e:
        logging.error(f"An error occurred in get_possible_files: {e}")
        print(f"An error occurred: {e}")
    return possible_files


def get_non_syllable_masker_file(masker_type):
    try:
        if masker_type == "pinknoise":
            pink_noise_DIR = DIR / "data" / "stim_files" / "pinknoise_resamp_24414"  # placeholder
            contents = [file for file in pink_noise_DIR.iterdir()]
            masker_file = os.path.join(DIR, get_random_file(contents))
        elif masker_type == "babble":
            babble_DIR = DIR / "data" / "stim_files" / "babble-numbers-reversed-n13-shifted_resamp_48828_resamp_24414"
            contents = [file for file in babble_DIR.iterdir()]
            masker_file = os.path.join(DIR, get_random_file(contents))
        else:
            masker_file = None
    except Exception as e:
        logging.error(f"An error occurred in get_non_syllable_masker_file: {e}")
        print(f"An error occurred: {e}")
        masker_file = None
    return masker_file


def get_random_file(files):
    try:
        return np.random.choice(files)
    except Exception as e:
        logging.error(f"An error occurred in get_random_file: {e}")
        print(f"An error occurred: {e}")


def get_target_number_file(sex=None, number=None, talker=None):
    try:
        stimuli = get_possible_files(sex=sex, number=number, talker=talker)
        target_file = get_random_file(stimuli)
        return target_file
    except Exception as e:
        logging.error(f"An error occurred in get_target_number_file: {e}")
        print(f"An error occurred: {e}")


def spacial_unmask_within_range(speaker_indices, target_speaker, sub_id, masker_type, stim_type, talker,
                                normalisation_method):
    global event_id
    block_id = 0
    try:
        print("Beginning spacial unmasking")
        iterator = speaker_indices
        np.random.shuffle(iterator)
        talker = talker
        trial_index = 0
        valid_responses = [1, 2, 3, 4, 5]
        stim_dir_list = [util.get_stim_dir(masker_type), util.get_stim_dir(stim_type)]
        max_n_samples = util.get_max_n_samples(stim_dir_list)

        for i in iterator:

            logging.info(f"Beginnign spatial unmasking at speaker with index {i}.")
            masking_speaker = freefield.pick_speakers(i)[0]
            stairs = slab.Staircase(start_val=-5, n_reversals=16, step_sizes=[7, 5, 3, 1])

            for level in stairs:
                logging.info(f"Presenting stimuli. this_trial_n = {stairs.this_trial_n}.")
                masker_file = get_non_syllable_masker_file(masker_type)
                target_file = get_target_number_file(talker=talker, number=np.random.randint(1, 6))
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
                    target_padded = np.pad(target.data, ((0, max_length - len(target.data)), (0, 0)), 'constant')
                    to_play_data = np.array(masker_padded) + np.array(target_padded)
                    to_play = slab.Sound(data=to_play_data)
                    util.set_multiple_signals(signals=[to_play], speakers=[target_speaker], equalize=False,
                                              max_n_samples=max_n_samples)
                else:
                    util.set_multiple_signals(signals=[target, masker], speakers=[target_speaker, masking_speaker],
                                              equalize=False, max_n_samples=max_n_samples)

                freefield.play(kind=1, proc="RX81")
                util.start_timer()

                response = None
                while True:
                    response = freefield.read("response", "RP2")
                    time.sleep(0.05)
                    if response in valid_responses:
                        break

                reaction_time = util.get_elapsed_time()

                if response == util.get_correct_response_number(target_file):
                    stairs.add_response(1)
                else:
                    stairs.add_response(0)

                save_per_response(event_id=event_id, sub_id=sub_id, block_id=block_id, trial_index=trial_index,
                                  step_number=stairs.this_trial_n, level_masker=masker.level,
                                  level_target=target.level, target_speaker=target_speaker,
                                  masking_speaker=masking_speaker, masker_filename=masker_file,
                                  target_filename=target_file,
                                  normalisation_method=normalisation_method,
                                  normalisation_level_masker=get_speaker_normalisation_level(masking_speaker),
                                  normalisation_level_target=get_speaker_normalisation_level(target_speaker),
                                  played_number=util.get_correct_response_number(target_file),
                                  response_number=response, reaction_time=reaction_time)
                trial_index += 1
                time.sleep(1.0)

            save_results(event_id=event_id, sub_id=sub_id, threshold=stairs.threshold(n=10),
                         distance_masker=masking_speaker.distance,
                         distance_target=target_speaker.distance, level_masker=masker.level,
                         level_target=target_speaker.level + stairs.threshold(n=10),
                         masker_type=masker_type, stim_type=stim_type, talker=talker,
                         normalisation_method=normalisation_method,
                         normalisation_level_masker=get_speaker_normalisation_level(masking_speaker),
                         normalisation_level_target=get_speaker_normalisation_level(target_speaker))
            block_id += 1
            logging.info(f"Block {block_id} completed. Ask questionnaire questions.")
            input("Press 'Enter' to continue with next experiment block.")
    except Exception as e:
        logging.error(f"An error occurred in spacial_unmask_within_range: {e}")
        print(f"An error occurred: {e}")


def get_key_by_value(value):
    try:
        return next((key for key, val in num_dict.items() if val == value), None)
    except Exception as e:
        logging.error(f"An error occurred in get_key_by_value: {e}")
        print(f"An error occurred: {e}")


def train_talker(talker_id):
    try:
        numbers = ["one", "two", "three", "four", "five", "six", "eight", "nine"]
        talker_num_dict = {}
        for number in numbers:
            filepath = os.path.join(DIR, get_possible_files(number=number, talker=talker_id)[0])
            talker_num_dict.update({number: filepath})
        print(talker_num_dict)
        for i in range(15):
            while not freefield.read("response", "RP2"):
                time.sleep(0.05)
            button_press = freefield.read("response", "RP2")
            button_press_written = get_key_by_value(button_press)
            print(button_press_written)
            print(talker_num_dict.get(button_press_written))
            signal = slab.Sound.read(talker_num_dict.get(button_press_written))
            signal = util.apply_mgb_equalization(signal=signal, speaker=freefield.pick_speakers(5)[0])
            freefield.set_signal_and_speaker(signal=signal, speaker=freefield.pick_speakers(5)[0], equalize=False)
            freefield.play(kind=1, proc="RX81")
            time.sleep(0.5)
            freefield.write(tag="data0", value=0, processors="RX81")
            freefield.write(tag="data1", value=0, processors="RX81")
        time.sleep(1.0)
    except Exception as e:
        logging.error(f"An error occurred in train_talker: {e}")
        print(f"An error occurred: {e}")


def set_event_id(new_event_id):
    global event_id
    event_id = new_event_id


def save_results(event_id, sub_id, threshold, distance_masker, distance_target,
                 level_masker, level_target, masker_type, stim_type, talker,
                 normalisation_method, normalisation_level_masker, normalisation_level_target):
    try:
        file_name = DIR / "data" / "results" / f"results_spacial_unmasking_{sub_id}.csv"

        if file_name.exists():
            df_curr_results = pd.read_csv(file_name)
        else:
            df_curr_results = pd.DataFrame()

        if len(level_masker.shape) == 1:
            level_masker = np.mean(level_masker)
        if len(level_target.shape) == 1:
            level_target = np.mean(level_target)

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

        new_row = {"event_id": event_id,
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
    except Exception as e:
        logging.error(f"An error occurred in save_results: {e}")
        print(f"An error occurred: {e}")


def save_per_response(event_id, sub_id, block_id, trial_index, step_number, level_masker, level_target,
                      target_speaker, masking_speaker, masker_filename, target_filename,
                      normalisation_method, normalisation_level_masker, normalisation_level_target, played_number,
                      response_number, reaction_time):
    try:
        is_correct = played_number == response_number

        file_name = DIR / "data" / "results" / f"results_per_step_spacial_unmasking_{sub_id}.csv"

        if file_name.exists():
            df_curr_results = pd.read_csv(file_name)
        else:
            df_curr_results = pd.DataFrame()

        if len(level_masker.shape) == 1:
            level_masker = np.mean(level_masker)
        if len(level_target.shape) == 1:
            level_target = np.mean(level_target)

        target_talker, target_sex, target_number = util.parse_country_or_number_filename(target_filename)

        new_row = {"event_id": None,
                   "timestamp": util.get_timestamp(),
                   "subject_id": sub_id,
                   "session_index": 3,
                   "plane": "distance",
                   "setup": "cathedral",
                   "task": "spatial_unmasking",
                   "block": block_id,
                   "trial_index": trial_index,
                   "headpose_offset_azi": 0,
                   "headpose_offset_ele": 0,
                   "target_number": target_number,
                   "target_filename": os.path.basename(target_filename),
                   "target_talker": target_talker,
                   "target_sex": target_sex,
                   "target_speaker_id": target_speaker.index,
                   "step_number:": step_number,
                   "distance_masker": masking_speaker.distance,
                   "distance_target": target_speaker.distance,
                   "target_speaker_proc": target_speaker.analog_proc,
                   "target_speaker_chan": target_speaker.analog_channel,
                   "target_speaker_azi": target_speaker.azimuth,
                   "target_speaker_ele": target_speaker.elevation,
                   "target_speaker_dist": target_speaker.distance,
                   "target_stim_level": level_target,
                   "masker_filename": os.path.basename(masker_filename),
                   "masker_speaker_id": masking_speaker.index,
                   "masker_speaker_proc": masking_speaker.analog_proc,
                   "masker_speaker_chan": masking_speaker.analog_channel,
                   "masker_speaker_azi": masking_speaker.azimuth,
                   "masker_speaker_ele": masking_speaker.elevation,
                   "masker_speaker_dist": masking_speaker.distance,
                   "masker_stim_level": level_masker,
                   "normalisation_method": normalisation_method,
                   "normalisation_level_masker": normalisation_level_masker,
                   "normalisation_level_target": normalisation_level_target,
                   "played_number": played_number,
                   "resp_number": response_number,
                   "is_correct": is_correct,
                   "reaction_time": reaction_time}

        df_curr_results = df_curr_results.append(new_row, ignore_index=True)
        df_curr_results.to_csv(file_name, mode='w', header=True, index=False)
    except Exception as e:
        logging.error(f"An error occurred in save_per_response: {e}")
        print(f"An error occurred: {e}")


def plot_target_ratio_vs_distance(sub_id, masker_type):
    try:
        data_file = DIR / "data" / "results" / f"results_spacial_unmasking_{sub_id}.csv"
        with open(data_file, 'rb') as f:
            results = pd.read_csv(f)

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
        fig.savefig(DIR / "data" / "results" / "figs" / f"results_{sub_id}.pdf")
    except Exception as e:
        logging.error(f"An error occurred in plot_target_ratio_vs_distance: {e}")
        print(f"An error occurred: {e}")


def plot_average_results(sub_ids="all"):
    try:
        result_filepath = DIR / "data" / "results"
        data_files = []
        results = pd.DataFrame()
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
                logging.error(f"An error occurred while reading file {file}: {e}")
                print(f"An error occurred: {e}")
                continue
            results = pd.concat([results, result], ignore_index=True)

        results["target_normalisation_adapted_ratio"] = (
                    results["level_target"] - results["normalisation_level_target"])

        grouped = pd.DataFrame(results).groupby("masker_type")
        grouped_babble = grouped.get_group("babble").groupby("distance_masker")
        average_tnr_babble = grouped_babble["target_normalisation_adapted_ratio"].mean()

        sns.pointplot(data=results, x="distance_masker", y="target_normalisation_adapted_ratio", errorbar="se")

        plt.xlabel("Distance of Masking Speaker")
        plt.ylabel("Ratio of Target Level")
        plt.title("Average Ratio of Target Level vs Distance of Masking Speaker")
        fig = plt.gcf()
        plt.show()
        plt.draw()
        fig.savefig(DIR / "data" / "results" / "figs" / f"average_results_.pdf")
    except Exception as e:
        logging.error(f"An error occurred in plot_average_results: {e}")
        print(f"An error occurred: {e}")


def get_speaker_normalisation_level(speaker, mgb_loudness=30):
    try:
        a, b, c = util.get_log_parameters(speaker.distance)
        return util.logarithmic_func(x=mgb_loudness, a=a, b=b, c=c)
    except Exception as e:
        logging.error(f"An error occurred in get_speaker_normalisation_level: {e}")
        print(f"An error occurred: {e}")

