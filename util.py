import freefield
import slab
import os
import pathlib
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime
import re
import logging
import ast
import statistics
import scipy
from scipy.stats import ttest_ind

"""import localisation

# Configure logging
import util"""

logging.basicConfig(filename='auditory_experiment.log', level=logging.ERROR)

DIR = pathlib.Path(os.curdir)

num_dict = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}

start_time = None

def initialize_setup():
    try:
        procs = [["RX81", "RX8", DIR / "data" / "rcx" / "cathedral_play_buf.rcx"],
                 ["RP2", "RP2", DIR / "data" / "rcx" / "button_numpad.rcx"]]
        freefield.initialize("cathedral", device=procs, zbus=False, connection="USB")
        freefield.SETUP = "cathedral"
        freefield.SPEAKERS = freefield.read_speaker_table()
        freefield.set_logger("INFO")
    except Exception as e:
        logging.error(f"An error occurred during setup initialization: {e}")
        print(f"An error occurred: {e}")

def quadratic_func(x, a, b, c):
    return a * x ** 2 + b * x + c

def logarithmic_func(x, a, b, c):
    try:
        return a * np.log(b * x) + c
    except Exception as e:
        logging.error(f"An error occurred in logarithmic_func: {e}")
        print(f"An error occurred: {e}")


def get_log_parameters(distance):
    try:
        parameters_file = DIR / "data" / "mgb_equalization_parameters" / "logarithmic_function_parameters.csv"
        parameters_df = pd.read_csv(parameters_file)
        params = parameters_df[parameters_df['speaker_distance'] == distance]
        a, b, c = params.iloc[0][['a', 'b', 'c']]
        return a, b, c
    except Exception as e:
        logging.error(f"An error occurred in get_log_parameters: {e}")
        print(f"An error occurred: {e}")

def get_quad_parameters(distance):
    try:
        parameters_file = DIR / "data" / "mgb_equalization_parameters" / "quadratic_function_parameters.csv"
        parameters_df = pd.read_csv(parameters_file)
        params = parameters_df[parameters_df['speaker_distance'] == distance]
        a, b, c = params.iloc[0][['a', 'b', 'c']]
        return a, b, c
    except Exception as e:
        logging.error(f"An error occurred in get_log_parameters: {e}")
        print(f"An error occurred: {e}")

def get_speaker_normalisation_level(speaker, mgb_loudness=30):
    try:
        a, b, c = get_log_parameters(speaker.distance)
        return logarithmic_func(x=mgb_loudness, a=a, b=b, c=c)
    except Exception as e:
        logging.error(f"An error occurred in get_speaker_normalisation_level: {e}")
        print(f"An error occurred: {e}")

def get_mgb_level_from_input_level(speaker, input_level):
    try:
        a, b, c = get_quad_parameters(speaker.distance)
        return quadratic_func(x=input_level, a=a, b=b, c=c)
    except Exception as e:
        logging.error(f"An error occurred in get_speaker_normalisation_level: {e}")
        print(f"An error occurred: {e}")

def get_mgb_level_from_input_level_and_distance(distance, input_level):
    try:
        a, b, c = get_quad_parameters(distance)
        return quadratic_func(x=input_level, a=a, b=b, c=c)
    except Exception as e:
        logging.error(f"An error occurred in get_mgb_level_from_input_level_and_distance: {e}")
        print(f"An error occurred: {e}")
def calculate_mgb_level_per_row(row):
    # Applies the calculation function to each row's target and masker distances and levels
    row['mgb_level_target'] = get_mgb_level_from_input_level_and_distance(row['distance_target'], row['level_target'])
    row['mgb_level_masker'] = get_mgb_level_from_input_level_and_distance(row['distance_masker'], row['level_masker'])
    return row


def get_stim_dir(stim_type):
    try:
        if stim_type == "babble":
            stim_dir = DIR / "data" / "stim_files" / "babble-numbers-reversed-n13-shifted_resamp_48828_resamp_24414"
        elif stim_type == "pinknoise":
            stim_dir = DIR / "data" / "stim_files" / "pinknoise_resamp_24414"
        elif stim_type == "countries" or stim_type == "countries_forward":
            stim_dir = DIR / "data" / "stim_files" / "tts-countries_n13_resamp_24414"
        elif stim_type == "countries_reversed":
            stim_dir = DIR / "data" / "stim_files" / "tts-countries-reversed_n13_resamp_24414"
        elif stim_type == "syllable":
            stim_dir = DIR / "data" / "stim_files" / "tts-numbers_n13_resamp_24414"
        elif stim_type == "uso":
            stim_dir = DIR / "data" / "stim_files" / "uso_300ms_resamp_24414"
        else:
            raise ValueError(f"Unknown stim_type: {stim_type}")
        return pathlib.Path(stim_dir)
    except Exception as e:
        logging.error(f"An error occurred in get_stim_dir: {e}")
        print(f"An error occurred: {e}")

def apply_mgb_equalization(signal, speaker, mgb_loudness=30, fluc=0):
    try:
        a, b, c = get_log_parameters(speaker.distance)
        signal.level = logarithmic_func(mgb_loudness + fluc, a, b, c)
        return signal
    except Exception as e:
        logging.error(f"An error occurred in apply_mgb_equalization: {e}")
        print(f"An error occurred: {e}")

def record_stimuli_and_create_csv(stim_type="countries_forward"):
    try:
        sounds_list, filenames_list = get_sounds_with_filenames(n="all", stim_type="stim_type")
        for i in range(len(sounds_list)):
            for speaker in freefield.SPEAKERS:
                sound = sounds_list[i]
                sound = apply_mgb_equalization(signal=sound, speaker=speaker, mgb_loudness=30)
                rec_sound = freefield.play_and_record(speaker=speaker, sound=sound, compensate_delay=True, equalize=False)
                sound_name = os.path.splitext(filenames_list[i])[0]
                filepath = DIR / "data" / "recordings" / f"{stim_type}" / f"sound-{sound_name}_distance-{speaker.distance}.wav"
                filepath = pathlib.Path(filepath)
                rec_sound.write(filename=filepath, normalise=False)
                time.sleep(2.7)
        # TODO: saving sound data to csv-file
    except Exception as e:
        logging.error(f"An error occurred in record_stimuli_and_create_csv: {e}")
        print(f"An error occurred: {e}")

def get_sounds_dict(stim_type="babble"):
    try:
        stim_dir = get_stim_dir(stim_type)
        sounds_dict = {}
        for file in stim_dir.iterdir():
            sounds_dict.update({str(file): slab.Sound.read(file)})
        return sounds_dict
    except Exception as e:
        logging.error(f"An error occurred in get_sounds_dict: {e}")
        print(f"An error occurred: {e}")

def get_sounds_with_filenames(sounds_dict, n="all", randomize=False):
    try:
        filenames_list = list(sounds_dict.values())
        sounds_list = list(sounds_dict.keys())
        if isinstance(n, int) or isinstance(n, np.int32):
            if randomize:
                random_indices = np.random.choice(len(sounds_list), n, replace=False)
                sounds_list = [sounds_list[i] for i in random_indices]
                filenames_list = [filenames_list[i] for i in random_indices]
            else:
                sounds_list = [sounds_list[i] for i in range(n)]
                filenames_list = [filenames_list[i] for i in range(n)]
        return sounds_list, filenames_list
    except Exception as e:
        logging.error(f"An error occurred in get_sounds_with_filenames: {e}")
        print(f"An error occurred: {e}")

def get_n_random_speakers(n):
    try:
        if n > len(freefield.SPEAKERS):
            raise ValueError("n cannot be greater than the length of the input list")
        random_indices = np.random.choice(len(freefield.SPEAKERS), n, replace=False)
        speakers = [freefield.pick_speakers(i)[0] for i in random_indices]
        return speakers, random_indices
    except Exception as e:
        logging.error(f"An error occurred in get_n_random_speakers: {e}")
        print(f"An error occurred: {e}")

def get_correct_response_number(file):
    try:
        for key in num_dict:
            if key in str(file):
                return num_dict.get(key)
    except Exception as e:
        logging.error(f"An error occurred in get_correct_response_number: {e}")
        print(f"An error occurred: {e}")

def create_localisation_config_file():
    try:
        config_dir = DIR / "config"
        filename = config_dir / "localisation_config.json"
        config_data = {"sounds_dict_pinknoise": get_sounds_dict("pinknoise"),
                       "sounds_dict_syllable": get_sounds_dict("syllable"),
                       "sounds_dict_babble": get_sounds_dict("babble")}
        with open(filename, 'w') as config_file:
            json.dump(config_data, config_file, indent=4)
    except Exception as e:
        logging.error(f"An error occurred in create_localisation_config_file: {e}")
        print(f"An error occurred: {e}")

def create_numerosity_judgement_config_file():
    try:
        config_dir = DIR / "config"
        filename = config_dir / "localisation_config.json"
        config_data = {"num_dict": {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six":  6, "seven": 7,
                                    "eight": 8, "nine": 9},
                       "talkers": ["p229", "p245", "p248", "p256", "p268", "p284", "p307", "p318"],
                       "n_sounds": [2, 3, 4, 5, 6]}
        with open(filename, 'w') as config_file:
            json.dump(config_data, config_file, indent=4)
    except Exception as e:
        logging.error(f"An error occurred in create_numerosity_judgement_config_file: {e}")
        print(f"An error occurred: {e}")

def read_config_file(experiment):
    try:
        if experiment == "localisation":
            filename = DIR / "config" / "localisation_config.json"
        elif experiment == "spacial_unmasking":
            filename = DIR / "config" / "spacial-unmasking_config.json"
        elif experiment == "numerosity_judgement":
            filename = DIR / "config" / "numerosity-judgement_config.json"
        else:
            raise ValueError(f"Unknown experiment: {experiment}")
        with open(filename, 'r') as config_file:
            config_data = json.load(config_file)
        return config_data
    except Exception as e:
        logging.error(f"An error occurred in read_config_file: {e}")
        print(f"An error occurred: {e}")

def set_multiple_signals(signals, speakers, equalize=True, mgb_loudness=30, fluc=0, max_n_samples=80000):
    try:
        freefield.write(tag="playbuflen", value=max_n_samples, processors="RX81")
        for i in range(len(signals)):
            if equalize:
                signals[i] = apply_mgb_equalization(signals[i], speakers[i], mgb_loudness, fluc)
            if not hasattr(speakers[i], 'index'):
                raise AttributeError(f"Speaker object at index {i} does not have an 'index' attribute.")
            speaker_chan = speakers[i].index + 1
            if len(signals[i].data.shape) == 1:
                data = np.pad(signals[i].data, (0, max_n_samples - len(signals[i].data)), 'constant')
            else:
                data = np.pad(signals[i].data, ((0, max_n_samples - len(signals[i].data)), (0, 0)), 'constant')
            if len(data.shape) == 2 and data.shape[1] == 2:
                data = np.mean(data, axis=1)
            freefield.write(tag=f"data{i}", value=data, processors="RX81")
            freefield.write(tag=f"chan{i}", value=speaker_chan, processors="RX81")
        time.sleep(0.1)
        for i in range(len(signals), 8):
            freefield.write(tag=f"chan{i}", value=99, processors="RX81")
            time.sleep(0.1)
        time.sleep(0.1)
    except Exception as e:
        logging.error(f"An error occurred in set_multiple_signals: {e}")
        print(f"An error occurred: {e}")

def test_speakers():
    try:
        sound = slab.Sound.pinknoise(duration=0.5, samplerate=24414)
        print(sound.samplerate)
        sound.level = 100
        for i in range(11):
            speaker = freefield.pick_speakers(i)[0]
            freefield.write(tag="playbuflen", value=len(sound), processors="RX81")
            freefield.write(tag="data0", value=sound.data, processors="RX81")
            freefield.write(tag="chan0", value=i+1, processors="RX81")
            time.sleep(2.0)
            freefield.play(kind=1, proc="RX81")
            time.sleep(2.0)
    except Exception as e:
        logging.error(f"An error occurred in test_speakers: {e}")
        print(f"An error occurred: {e}")

def start_timer():
    global start_time
    start_time = time.time()

def get_elapsed_time(reset=True):
    global start_time
    try:
        if start_time is None:
            raise ValueError("Timer has not been started.")
        elapsed_time = time.time() - start_time
        if reset:
            start_time = None
        return elapsed_time
    except Exception as e:
        logging.error(f"An error occurred in get_elapsed_time: {e}")
        print(f"An error occurred: {e}")

def get_timestamp():
    try:
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        logging.error(f"An error occurred in get_timestamp: {e}")
        print(f"An error occurred: {e}")

def parse_country_or_number_filename(filepath):
    try:
        pattern = r"talker-(?P<talker>.+?)_sex-(?P<sex>[.\w]+)_text-(?P<text>[.\w]+)\.wav"
        filename = str(os.path.basename(filepath))
        match = re.match(pattern, filename)
        if match:
            talker = match.group("talker")
            sex = match.group("sex")
            text = match.group("text")
            return talker, sex, text
        else:
            raise ValueError(f"Filename {filename} does not match the file pattern")
    except Exception as e:
        logging.error(f"An error occurred in parse_country_or_number_filename: {e}")
        print(f"An error occurred: {e}")

def create_resampled_stim_dirs(samplerate=24414):
    try:
        stim_files_dir = DIR / "data" / "stim_files"
        for stim_dir in stim_files_dir.iterdir():
            pattern = r"(?P<stim_type>.+?)_n13_resamp_(?P<samplerate>.+?)"
            match = re.match(pattern, str(os.path.basename(stim_dir)))
            if match:
                stim_type = match.group("stim_type")
                if not match.group("samplerate") == samplerate:
                    new_dir_path = stim_files_dir / f"{stim_type}_n13_resamp_{samplerate}"
                    if not os.path.exists(new_dir_path):
                        os.makedirs(new_dir_path)
                    for file in stim_dir.iterdir():
                        sound = slab.Sound.read(str(file))
                        sound = sound.resample(samplerate=samplerate)
                        new_filepath = new_dir_path / os.path.basename(file)
                        sound.write(new_filepath)
            elif stim_dir.is_dir():
                new_dir_name = str(os.path.basename(stim_dir)) + f"_resamp_{samplerate}"
                new_dir_path = stim_files_dir / new_dir_name
                if not os.path.exists(new_dir_path):
                    os.makedirs(new_dir_path)
                for file in stim_dir.iterdir():
                    sound = slab.Sound.read(str(file))
                    sound = sound.resample(24414)
                    new_filepath = new_dir_path / os.path.basename(file)
                    sound.write(new_filepath)
    except Exception as e:
        logging.error(f"An error occurred in create_resampled_stim_dirs: {e}")
        print(f"An error occurred: {e}")

def initialize_stim_recording():
    try:
        freefield.set_logger("DEBUG")
        freefield.SETUP = "cathedral"
        freefield.initialize(setup="cathedral", default="play_birec", connection="USB", zbus=False)
        freefield.SETUP = "cathedral"
        freefield.SPEAKERS = freefield.read_speaker_table()
    except Exception as e:
        logging.error(f"An error occurred in initialize_stim_recording: {e}")
        print(f"An error occurred: {e}")

def get_max_n_samples(stim_dirs):
    try:
        sounds_list = []
        if not isinstance(stim_dirs, list):
            stim_dirs = [stim_dirs]
        for directory in stim_dirs:
            directory_path = pathlib.Path(directory)
            if directory_path.is_dir():
                for file in directory_path.iterdir():
                    if file.is_file() and file.suffix == '.wav':
                        try:
                            sounds_list.append(slab.Sound.read(file))
                        except Exception as e:
                            logging.error(f"Error reading file {file}: {e}")
                            print(f"Error reading file {file}: {e}")
            else:
                raise ValueError(f"{directory} is not a directory")
        max_n_samples = max(sound.n_samples for sound in sounds_list)
        return max_n_samples
    except Exception as e:
        logging.error(f"An error occurred in get_max_n_samples: {e}")
        print(f"An error occurred: {e}")

def record_stimuli(stim_type, mgb_loudness=27.5):
    try:
        stim_dir = get_stim_dir(stim_type)
        recording_dir = DIR / "data" / "recordings" / stim_type
        if not os.path.exists(recording_dir):
            os.makedirs(recording_dir)
        for file in stim_dir.iterdir():
            sound = slab.Sound.read(file)
            print(sound.samplerate)
            print(sound.n_samples)
            if len(sound.data.shape) == 2 and sound.data.shape[1] == 2:
                data = np.mean(sound.data, axis=1)
                sound = slab.Sound(data, samplerate=sound.samplerate)
                print(sound.samplerate)
                print(sound.n_samples)
            basename = str(os.path.splitext(os.path.basename(file))[0])
            for speaker in freefield.SPEAKERS:
                sound = apply_mgb_equalization(sound, speaker, mgb_loudness, 0)
                recording = freefield.play_and_record(speaker=speaker, sound=sound, compensate_delay=True, equalize=False)
                print(recording.n_samples)
                print(recording.samplerate)
                recording_filename = f"sound-{basename}_mgb-level-{mgb_loudness}_distance-{speaker.distance}.wav"
                recording_filepath = recording_dir / recording_filename
                recording.write(filename=recording_filepath, normalise=False)
                time.sleep(1.0)
    except Exception as e:
        logging.error(f"An error occurred in record_stimuli: {e}")
        print(f"An error occurred: {e}")

def reverse_stimuli(stim_type):
    try:
        stim_dir = get_stim_dir(stim_type)
        pattern = r"(?P<stim_type>.+?)_n13_resamp_(?P<samplerate>.+)"
        match = re.match(pattern, str(os.path.basename(stim_dir)))
        if match:
            samplerate = match.group("samplerate")
            print(samplerate)
            target_dir_name = f"{stim_type}_reversed_n13_resamp_{samplerate}"
        else:
            target_dir_name = str(os.path.basename(stim_dir)) + "_reversed"
        target_dir = DIR / "data" / "stim_files" / target_dir_name
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        for file in stim_dir.iterdir():
            basename = os.path.splitext(os.path.basename(file))[0]
            sound = slab.Sound.read(file)
            new_data = sound.data[::-1]
            sound.data = new_data
            filename = target_dir / f"{basename}_reversed.wav"
            sound.write(filename=filename)
    except Exception as e:
        logging.error(f"An error occurred in reverse_stimuli: {e}")
        print(f"An error occurred: {e}")

def add_dynamic_range_to_num_judge(stim_type, sub_id):
    filepath = DIR / "data" / "results" / f"results_numerosity_judgement_{stim_type}_{sub_id}.csv"
    df = pd.read_csv(filepath)
    dynamic_range_df = 0000
    p_ref = 2e-5
    upper_freq = 11000

    for trial in df["trial_index"]:
        row = df[df["trial_index"] == trial]
        if not row.empty:
            stim_number = row["stim_number"].values[0]
            speaker_ids = ast.literal_eval(row["speaker_ids"].values[0])
            stim_country_ids = ast.literal_eval(row["stim_country_ids"].values[0])
            stim_talker_ids = ast.literal_eval(row["stim_country_ids"].values[0])
            stim_sexes = ast.literal_eval(row["stim_sexes"].values[0])
            rec_list = list()
            for i in range(stim_number):
                speaker_distance = freefield.pick_speakers(speaker_ids[i])[0].distance
                recording_filename = f"sound-talker-{stim_talker_ids[i]}_sex-{stim_sexes[i]}_text-" \
                                     f"{stim_country_ids[i]}_mgb-level-30_distance-{speaker_distance}.wav"
                recording_path = DIR / "data" / "recordings" / f"{stim_type}" / recording_filename
                rec = slab.Sound.read(str(recording_path))
                rec_list.append(rec)

            summed_data = None
            for rec in rec_list:
                summed_data += rec.data

            summed_sound = slab.Sound(summed_data)
            freqs, times, power = summed_sound.spectrogram(show=False)

            power = power[freqs < upper_freq, :]
            power = 10 * np.log10(power / (p_ref ** 2))  # logarithmic power for plotting
            dB_max = power.max()
            dB_min = dB_max - 65 # value with the highest variance according to max
            interval = power[np.where((power > dB_min) & (power < dB_max))]
            percentage_filled = interval.shape[0] / power.flatten().shape[0]
            df.loc[df["trial_index"] == trial, "dyn_range_percentage_filled"] = percentage_filled


def get_spectral_coverage(filenames, speaker_ids, stim_type, trial_dur=0.6):
    trial_composition = list()
    p_ref = 2e-5  # 20 μPa, the standard reference pressure for sound in air
    upper_freq = 11000  # upper frequency limit that carries information for speech
    dyn_range = 65
    for i in len(filenames):
        talker, sex, text = parse_country_or_number_filename(filenames[i])
        speaker = freefield.pick_speakers(speaker_ids[i])[0]
        rec_name = f"sound-talker-{talker}_sex-{sex}_text-{text}_mgb-level-30_distance-{speaker.distance}.wav"
        rec_dir = None
        if stim_type == "countries_forward":
            rec_dir = DIR / "data" / "recordings" / "countries_forward"
        elif stim_type == "countries_reversed":
            rec_dir = DIR / "data" / "recordings" / "countries_reversed"
        stim = slab.Sound.read(os.path.join(rec_dir, rec_name))
        trial_composition.append(stim)
    sound = sum(trial_composition)
    sound = slab.Sound(sound.data.mean(axis=1), samplerate=sound.samplerate)
    sound = sound.resample(24414)
    freqs, times, power = sound.spectrogram(show=False)
    power = 10 * np.log10(power / (p_ref ** 2))  # logarithmic power for plotting
    power = power[freqs < upper_freq, :]
    dB_max = power.max()
    dB_min = dB_max - dyn_range
    interval = power[np.where((power > dB_min) & (power < dB_max))]
    percentage_filled = interval.shape[0] / power.flatten().shape[0]
    return percentage_filled

def parse_staircases(filename, only_index=False):
    # Define the regex pattern
    pattern = r'speaker_index-(\d+)_sub_id-sub_(\d+)'

    # Search for the pattern in the filename
    match = re.search(pattern, filename)

    # Check if the pattern was found
    if match:
        # Extract speaker_index and sub_id
        speaker_index = int(match.group(1))
        sub_id = int(match.group(2))
        if only_index:
            return speaker_index
        else:
            return speaker_index, sub_id
    else:
        raise ValueError(f"Filename '{filename}' does not match the expected pattern.")

def get_speaker_index(filename):
    # Use regex to find the speaker_index in the filename
    match = re.search(r'speaker_index-(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        return float('inf')  # Return infinity if the pattern is not found to handle unexpected filenames

def plot_subject_staircases(sub_id, draw=True):
    directory_path = DIR / "data" / "results"
    file_list = list()
    df_path = DIR / "data" / "results" / f"results_spacial_unmasking_{sub_id}.csv"
    df = pd.read_csv(df_path)
    index_order = df["distance_masker"].tolist()
    index_order = [int(distance - 2.0) for distance in index_order]
    fig, axes = plt.subplots(2, 4, figsize=(14, 10))
    for file in directory_path.iterdir():
        for index in index_order:
            if file.is_file() and file.suffix == '.json' and str(sub_id) in str(file) and f"index-{index}" in str(file):
                try:
                    file_list.append(str(file))
                except Exception as e:
                    logging.error(f"Error reading file {file}: {e}")
                    print(f"Error reading file {file}: {e}")

    """file_list = sorted(file_list, key=get_speaker_index)"""

    for i in range(len(file_list)):
        with open(file_list[i], 'r') as file:
            data = json.load(file)

        # Extract the "intensities" array
        intensities = data.get("intensities", [])
        reversal_intensities = data.get("reversal_intensities", [])
        threshold = sum(reversal_intensities[-8:]) / 8
        print(intensities)
        print(len(intensities))
        print(list(range(len(intensities))))

        filename = str(file_list[i])
        speaker_index, sub_id = parse_staircases(filename)

        if i < 4:
            sns.lineplot(x=list(range(len(intensities))), y=intensities, ax=axes[0, i])
            axes[0, i].axhline(y=threshold, color='red', linestyle='--', label='threshold')
            axes[0, i].set_title(f'speaker_index = {speaker_index}')
            axes[0, i].set_xlabel('trial_n')
            axes[0, i].set_ylabel('threshold')
            axes[0, i].set_ylim(-25, 5)
        else:
            sns.lineplot(x=list(range(len(intensities))), y=intensities, ax=axes[1, (i - 4)])
            axes[1, (i - 4)].axhline(y=threshold, color='red', linestyle='--', label='threshold')
            axes[1, (i - 4)].set_title(f'speaker_index = {speaker_index}')
            axes[1, (i - 4)].set_xlabel('trial_n')
            axes[1, (i - 4)].set_ylabel('threshold')
            axes[1, (i - 4)].set_ylim(-25, 5)

    # Adjust the layout
    plt.tight_layout()

    plt.savefig(DIR / "data" / "results" / "figs" / f"fig_spatial_unmasking_staircases_sub_id-{sub_id}.jpeg")

    # Display the plots
    if draw:
        plt.show()

def plot_learning_effect_per_trial(task="numerosity_judgement"):
    directory_path = DIR / "data" / "results"
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    df = pd.DataFrame()

    for file in directory_path.iterdir():
        if task == "numerosity_judgement":
            if file.is_file() and "results_numerosity_judgement" in str(file) and file.suffix == ".csv":
                this_df = pd.read_csv(file)
                df = pd.concat([df, this_df], ignore_index=True)
        if task == "localisation_accuracy":
            if file.is_file() and "results_localisation_accuracy" in str(file) and file.suffix == ".csv":
                this_df = pd.read_csv(file)
                df = pd.concat([df, this_df], ignore_index=True)

    if task == "localisation_accuracy":
        df["abs_error"] = abs(df["stim_dist"] - df["resp_dist"])
    elif task == "numerosity_judgement":
        df["abs_error"] = abs(df["stim_number"] - df["resp_number"])

    df_blocks = [df[df["block"] == i] for i in range(1,5)]

    for i, ax in enumerate(axes):
        if i < len(df_blocks):  # Ensure we have enough blocks to plot
            grouped_by_trial_index = df_blocks[i].groupby("trial_index")
            """if task == "numerosity_judgement":
                hit_rates = grouped_by_trial_index["is_correct"].mean().reset_index()
                sns.scatterplot(x='trial_index', y='is_correct', data=hit_rates, ax=ax)
                sns.regplot(x='trial_index', y='is_correct', data=hit_rates, ax=ax, scatter=False, ci=None, color='r')
                ax.set_title(f'Block {i + 1}')
                ax.set_xlabel('Trial Index')
                ax.set_ylabel('Mean Hit Rate')
                ax.set_ylim(0, 1)  # Assuming hit rate is between 0 and 1"""
            average_abs_error = grouped_by_trial_index["abs_error"].mean().reset_index()
            sns.scatterplot(x='trial_index', y='abs_error', data=average_abs_error, ax=ax)
            sns.regplot(x='trial_index', y='abs_error', data=average_abs_error, ax=ax, scatter=False, ci=None, color='r')
            ax.set_title(f'Block {i + 1}')
            ax.set_xlabel('Trial Index')
            ax.set_ylabel('Mean Absolute Error')
            ax.set_ylim(0, 5)
        else:
            fig.delaxes(ax)  # Remove any unused axes

    plt.tight_layout()
    plt.savefig(DIR / "data" / "results" / "figs" / f"fig_learning_effect_by_trial_{task}.jpeg")
    plt.show()

def plot_numerosity_judgement_distance_dependencies(dependency, num_bins=30):
    directory_path = DIR / "data" / "results"
    df = pd.DataFrame()
    for file in directory_path.iterdir():
        if file.is_file() and "results_numerosity_judgement" in str(file) and file.suffix == ".csv":
            this_df = pd.read_csv(file)
            df = pd.concat([df, this_df], ignore_index=True)
    df["abs_error"] = abs(df["stim_number"] - df["resp_number"])
    df['binned_dependency'] = pd.cut(df[dependency], bins=num_bins)
    grouped_df = df.groupby(['binned_dependency', 'stim_number']).agg({
        'resp_number': 'mean',
        dependency: 'mean'
    }).reset_index()

    """sns.scatterplot(grouped_df, x=dependency, y="abs_error")
    sns.regplot(x=dependency, y="abs_error", data=grouped_df, scatter=False, ci=None, color='r')"""
    sns.scatterplot(data=grouped_df, x=dependency, y='resp_number', hue='stim_number', palette='viridis')

    for stim_num in grouped_df['stim_number'].unique():
        sns.regplot(
            x=grouped_df[grouped_df['stim_number'] == stim_num][dependency],
            y=grouped_df[grouped_df['stim_number'] == stim_num]['resp_number'],
            scatter=False, ci=None, color='r', label=f'Trend (stim {stim_num})'
        )
    plt.tight_layout()
    plt.savefig(DIR / "data" / "results" / "figs" / f"fig_numerosity_judgement_{dependency}_effect.jpeg")
    plt.show()
    return

def get_experiment_order(sub_id):
    directory_path = DIR / "data" / "results"
    df_spatial_unmasking = pd.read_csv(pathlib.Path(directory_path / f"results_per_step_spacial_unmasking_{sub_id}.csv"))
    df_numerosity_judgement = pd.read_csv(pathlib.Path(directory_path / f"results_numerosity_judgement_{sub_id}.csv"))
    df_localisation_accuracy = pd.read_csv(pathlib.Path(directory_path / f"results_localisation_accuracy_{sub_id}.csv"))

    result_df_dict = {"spatial_unmasking": df_spatial_unmasking, "numerosity_judgement": df_numerosity_judgement, "localisation_accuracy": df_localisation_accuracy}
    for key in result_df_dict.keys():
        df = result_df_dict.get(key)
        timestamp = df.iloc[0]["timestamp"]
        result_df_dict.update({key: timestamp})

    sorted_experiment_list = dict(sorted(result_df_dict.items(), key=lambda item: item[1])).keys()
    return sorted_experiment_list

def get_stimulus_order(sub_id, task):
    directory_path = DIR / "data" / "results"
    if task == "localisation_accuracy":
        df = pd.read_csv(pathlib.Path(directory_path / f"results_localisation_accuracy_{sub_id}.csv"))
    elif task == "numerosity_judgement":
        df = pd.read_csv(pathlib.Path(directory_path / f"results_numerosity_judgement{sub_id}.csv"))

    stim_types = df["stim_type"].unique().tolist()
    stim_types_dict = {}
    for stim_type in stim_types:
        filtered_df =df[df["stim_type"] == stim_type]
        stim_types_dict.update({stim_type: min(filtered_df["timestamp"].tolist())})

    sorted_stim_type_list = dict(sorted(stim_types_dict.items(), key=lambda item: item[1])).keys()
    return sorted_stim_type_list

def plot_task_order_effects(task):
    directory_path = DIR / "data" / "results"
    df_list = list()
    for file in directory_path.iterdir():
        if task == "numerosity_judgement":
            if file.is_file() and "results_numerosity_judgement" in str(file) and file.suffix == ".csv":
                this_df = pd.read_csv(file)
                df_list.append(this_df)

        elif task == "localisation_accuracy":
            if file.is_file() and "results_localisation_accuracy" in str(file) and file.suffix == ".csv":
                this_df = pd.read_csv(file)
                df_list.append(this_df)

    """df_first_pos = pd.DataFrame()
    df_second_pos = pd.DataFrame()
    df_third_pos = pd.DataFrame()"""
    complete_df = pd.DataFrame()

    for df in df_list:
        sub_id = df["subject_id"].unique().tolist()[0]
        print(sub_id)
        sorted_experiment_list = list(get_experiment_order(sub_id))
        if task == sorted_experiment_list[0]:
            df["experiment_order"] = 0
        elif task == sorted_experiment_list[1]:
            df["experiment_order"] = 2
        elif task == sorted_experiment_list[2]:
            df["experiment_order"] = 3
        complete_df = pd.concat([complete_df, df])

    if task == "localisation_accuracy":
        complete_df["error"] = complete_df["stim_dist"] - complete_df["resp_dist"]
    elif task == "numerosity_judgement":
        complete_df["error"] = complete_df["stim_number"] - complete_df["resp_number"]

    sns.boxplot(complete_df, x="experiment_order", y="error")

    plt.tight_layout()
    plt.savefig(DIR / "data" / "results" / "figs" / f"fig_{task}_task_order_effect.jpeg")
    plt.show()

    return

def plot_stimulus_effects(task):
    directory_path = DIR / "data" / "results"
    df = pd.DataFrame()
    for file in directory_path.iterdir():
        if file.is_file() and f"results_{task}" in str(file) and file.suffix == ".csv":
            this_df = pd.read_csv(file)
            df = pd.concat([df, this_df], ignore_index=True)

    if task == "localisation_accuracy":
        df["error"] = df["stim_dist"] - df["resp_dist"]
        sns.boxplot(df, x="stim_dist", y="error", hue="stim_type")
    elif task == "numerosity_judgement":
        df["error"] = df["stim_number"] - df["resp_number"]
        sns.boxplot(df, x="stim_number", y="error", hue="stim_type")
    """sns.boxplot(df, x="stim_type", y="error")"""

    plt.tight_layout()
    plt.savefig(DIR / "data" / "results" / "figs" / f"fig_{task}_stim_type_effect.jpeg")
    plt.show()
    return

def get_spatial_unmasking_performance(sub_id):
    return

def get_localisation_accuracy_performance(sub_id):
    return

def create_summed_result_file(task):
    dir = DIR / "data" / "results"
    df = pd.DataFrame()
    filename = DIR / "data" / "results" / "added_result_files" / f"{task}_summed.csv"
    for file in dir.iterdir():
        if file.is_file() and f"results_{task}" in str(file) and file.suffix == ".csv":
            sub_df = pd.read_csv(file)
            df = pd.concat([sub_df, df], ignore_index=True)


    df.to_csv(filename, mode='w', header=True, index=False)
def get_su_slope_closest_speakers(subject_id):
    file = DIR / "data" / "results" / "added_result_files" / "spacial_unmasking_summed_and_processed.csv"
    df = pd.read_csv(file)
    df_filtered = df[df["subject"] == subject_id]

    filtered_row_colocated = df_filtered[df_filtered["abs_spatial_separation"] == 0.0]
    filtered_rows_neighbors = df_filtered[df_filtered["abs_spatial_separation"] == 1.0]
    print(subject_id)
    print(filtered_rows_neighbors)
    print(filtered_rows_neighbors["threshold"].values)
    x_colocated = 0
    y_colocated = filtered_row_colocated["threshold"]
    x_neighbors = 1
    y_neighbors = statistics.mean(filtered_rows_neighbors["threshold"].values)
    slope = float(y_neighbors - y_colocated) / (x_neighbors - x_colocated)
    return slope

def get_su_tmr_slope(subject_id):
    file = DIR / "data" / "results" / "added_result_files" / "spacial_unmasking_summed_and_processed.csv"
    df = pd.read_csv(file)

    subject_data = df[df['subject'] == subject_id]
    X = subject_data['abs_spatial_separation'].values
    y = subject_data['relative_tmr'].values

    # Fit linear regression using numpy's polyfit (degree 1 for linear)
    slope_tmr, intercept = np.polyfit(X, y, 1)
    return slope_tmr
def process_nj_data():
    file = "/Users/lukaslange/PycharmProjects/spacial_unmasking/data/results/added_result_files/numerosity_judgement.csv"
    df = pd.read_csv(file)
    df_filtered = df[df["plane"] == "distance"]
    target_filename = DIR / "data" / "results" / "added_result_files" / f"numerosity_judgement_processed.csv"
    # Initialize an empty column for the slopes
    df_filtered["su_slope_closest_speaker"] = None
    df_filtered["su_tmr_slope"] = None
    df_filtered["la_rmse_pinknoise"] = None
    df_filtered["la_rmse_babble"] = None
    df_filtered["la_mae_pinknoise"] = None
    df_filtered["la_mae_babble"] = None
    df_filtered["la_r2"] = None

    # Iterate over each unique subject_id
    for subject_id in df_filtered["subject_id"].unique():
        # Calculate the slope for the current subject_id
        slope = get_su_slope_closest_speakers(subject_id)
        slope_tmr = get_su_tmr_slope(subject_id)
        la_rmse_pinknoise, la_rmse_babble = get_la_rmse_per_stimtype(subject_id)
        la_mae_pinknoise, la_mae_babble = get_la_mae_per_stimtype(subject_id)
        la_r2 = get_r2_from_rsme(subject_id)
        print(la_r2)
        # Assign the slope to the respective rows in the new column
        df_filtered.loc[df_filtered["subject_id"] == subject_id, "su_slope_closest_speaker"] = slope
        df_filtered.loc[df_filtered["subject_id"] == subject_id, "su_parameter"] = (df_filtered.loc[df_filtered["subject_id"] == subject_id, "su_slope"] + slope) / 2
        df_filtered.loc[df_filtered["subject_id"] == subject_id, "la_rmse_pinknoise"] = la_rmse_pinknoise
        df_filtered.loc[df_filtered["subject_id"] == subject_id, "la_rmse_babble"] = la_rmse_babble
        df_filtered.loc[df_filtered["subject_id"] == subject_id, "la_mae_pinknoise"] = la_mae_pinknoise
        df_filtered.loc[df_filtered["subject_id"] == subject_id, "la_mae_babble"] = la_mae_babble
        df_filtered.loc[df_filtered["subject_id"] == subject_id, "la_r2"] = la_r2
        df_filtered.loc[df_filtered["subject_id"] == subject_id, "su_slope_closest_speaker"] = slope_tmr
    df_filtered.to_csv(target_filename, mode='w', header=True, index=False)

def process_spatial_unmasking_data():
    file = "/Users/lukaslange/PycharmProjects/spacial_unmasking/data/results/added_result_files/spacial_unmasking_summed.csv"
    df = pd.read_csv(file)
    df["abs_spatial_separation"] = abs(df["distance_target"] - df["distance_masker"])
    df['relative_threshold'] = df.groupby('subject')['threshold'].transform(lambda x: x - x[df['distance_masker'] == 7].iloc[0])
    df = df.apply(calculate_mgb_level_per_row, axis=1)
    df['tmr'] = (df["mgb_level_target"] / df["mgb_level_masker"])
    df['relative_tmr'] = df.groupby('subject')['tmr'].transform(lambda x: x - x[df['distance_masker'] == 7].iloc[0])
    filename = DIR / "data" / "results" / "added_result_files" / f"spacial_unmasking_summed_and_processed.csv"
    df.to_csv(filename, mode='w', header=True, index=False)

def plot_spatial_unmasking_individual_data():
    file = "/Users/lukaslange/PycharmProjects/spacial_unmasking/data/results/added_result_files/spacial_unmasking_summed_and_processed.csv"
    df = pd.read_csv(file)

    plt.figure(figsize=(8, 6))

    sns.lineplot(df, x="distance_masker", y="relative_threshold", hue="subject")

    # Place the legend to the right of the plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Subjects")

    plt.xlabel("distance of masking speaker in m")
    plt.ylabel("relative threshold in dB")

    plt.tight_layout()
    plt.savefig(DIR / "data" / "results" / "figs" / "fig_spatial_unmasking_individual_data.jpeg")

def plot_spatial_unmasking_data():
    file = "/Users/lukaslange/PycharmProjects/spacial_unmasking/data/results/added_result_files/spacial_unmasking_summed_and_processed.csv"
    df = pd.read_csv(file)

    # Perform t-tests comparing distance_masker = 7 with 6 and 8
    df_2 = df[df["distance_masker"] == 2]["relative_tmr"]
    df_4 = df[df["distance_masker"] == 4]["relative_tmr"]
    df_6 = df[df["distance_masker"] == 6]["relative_tmr"]
    df_7 = df[df["distance_masker"] == 7]["relative_tmr"]
    df_8 = df[df["distance_masker"] == 8]["relative_tmr"]
    df_10 = df[df["distance_masker"] == 10]["relative_tmr"]
    df_12 = df[df["distance_masker"] == 12]["relative_tmr"]

    t_stat_4_vs_7, p_val_4_vs_7 = ttest_ind(df_4, df_7)
    t_stat_6_vs_7, p_val_6_vs_7 = ttest_ind(df_6, df_7)
    t_stat_8_vs_7, p_val_8_vs_7 = ttest_ind(df_8, df_7)
    t_stat_10_vs_12, p_val_10_vs_12 = ttest_ind(df_10, df_12)
    t_stat_2_vs_7, p_val_2_vs_7 = ttest_ind(df_2, df_7)
    t_stat_7_vs_10, p_val_7_vs_10 = ttest_ind(df_7, df_10)
    t_stat_7_vs_12, p_val_7_vs_12 = ttest_ind(df_7, df_12)

    plt.figure(figsize=(8, 6))

    grouped_df = df.groupby("distance_masker").agg(
        mean_relative_tmr=("relative_tmr", "mean"),
        std_relative_tmr=("relative_tmr", "std")
    ).reset_index()

    # Plot the mean points
    sns.lineplot(data=grouped_df, x="distance_masker", y="mean_relative_tmr",
                 marker="o")

    # Add error bars
    plt.errorbar(grouped_df["distance_masker"], grouped_df["mean_relative_tmr"],
                yerr=grouped_df["std_relative_tmr"], fmt='o', color='blue', capsize=5)

    for i in range(len(grouped_df)):
        x_value = grouped_df.loc[i, "distance_masker"]
        mean_value = grouped_df.loc[i, "mean_relative_tmr"]
        sem_value = grouped_df.loc[i, "std_relative_tmr"]
        print(str(x_value) + ", " + str(mean_value) + ", " + str(sem_value))

    # Add annotations for significance
    if p_val_4_vs_7 < 0.05:
        plt.text(4.1, df_4.mean(), '*', ha='center', va='bottom', color='black', fontsize=20)
    if p_val_6_vs_7 < 0.05:
        plt.text(6.1, df_6.mean(), '*', ha='center', va='bottom', color='black', fontsize=20)
    if p_val_8_vs_7 < 0.05:
        plt.text(8.1, df_8.mean(), '*', ha='center', va='bottom', color='black', fontsize=20)
    if p_val_10_vs_12 < 0.05:
        plt.text(11.1, -2.25 , '*', ha='center', va='bottom', color='black', fontsize=20)
    if p_val_2_vs_7 < 0.05:
        plt.text(2.1, -1.9 , '*', ha='center', va='bottom', color='black', fontsize=20)
    if p_val_7_vs_12 < 0.05:
        plt.text(12.1, df_12.mean(), '*', ha='center', va='bottom', color='black', fontsize=20)
    if p_val_7_vs_10 < 0.05:
        plt.text(10.1, df_10.mean(), '*', ha='center', va='bottom', color='black', fontsize=20)


    plt.xlabel("distance of masking speaker in m")
    plt.ylabel("relative TMR")
    plt.tight_layout()
    plt.savefig(DIR / "data" / "results" / "figs" / "fig_spatial_unmasking_data_tmr.pdf")
    plt.show()
def plot_localisation_individual_data():
    file = "/Users/lukaslange/PycharmProjects/spacial_unmasking/data/results/added_result_files/localisation_accuracy_summed.csv"
    df = pd.read_csv(file)
    plt.figure(figsize=(8, 6))

    sns.lineplot(df, x="stim_dist", y="resp_dist", hue="subject_id", errorbar = None)

    # Add a black line with slope = 1 and intercept = 0
    x_vals = np.array(plt.gca().get_xlim())  # Get current x-axis limits
    y_vals = x_vals  # Since slope = 1 and intercept = 0, y = x
    plt.plot(x_vals, y_vals, '--', color='black')

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Subjects")
    plt.gca().set_aspect('equal', adjustable='box')

    plt.xlabel("stimulus distance in m")
    plt.ylabel("estimated distance in m")

    plt.tight_layout()
    plt.savefig(DIR / "data" / "results" / "figs" / "fig_localisation_individual_data.jpeg")
    plt.show()


def plot_localisation_slopes():
    file = "/Users/lukaslange/PycharmProjects/spacial_unmasking/data/results/added_result_files/localisation_accuracy_summed.csv"
    df = pd.read_csv(file)
    plt.figure(figsize=(9, 6))

    """sns.scatterplot(df, x="stim_dist", y="resp_dist", hue="subject_id")"""

    # Fit and plot a regression line for each subject
    subjects = df['subject_id'].unique()

    for subject in subjects:
        subject_data = df[df['subject_id'] == subject]
        X = subject_data['stim_dist'].values
        y = subject_data['resp_dist'].values

        # Fit linear regression using numpy's polyfit (degree 1 for linear)
        slope, intercept = np.polyfit(X, y, 1)
        print("slope = " + str(slope))
        print("intercept = " + str(intercept))

        # Calculate the regression line only within the range of the data
        x_vals = np.linspace(X.min(), X.max(), 100)
        y_vals = slope * x_vals + intercept

        # Plot regression line without adding to the legend
        plt.plot(x_vals, y_vals, linestyle='-', linewidth=2, color='gray', alpha=0.7)

        # Plot regression line
        plt.plot(x_vals, y_vals, label=f'Regression: Subject {subject}', linestyle='-', linewidth=2)
    # Add a black line with slope = 1 and intercept = 0

    x_vals = np.array(plt.gca().get_xlim())  # Get current x-axis limits
    y_vals = x_vals  # Since slope = 1 and intercept = 0, y = x
    plt.plot(x_vals, y_vals, '--', color='black')

    # Legend only for the subjects (not the regression lines)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Subjects")

    plt.gca().set_aspect('equal', adjustable='box')

    plt.xlabel("stimulus distance in m")
    plt.ylabel("estimated distance in m")

    plt.tight_layout()
    plt.savefig(DIR / "data" / "results" / "figs" / "fig_localisation_individual_data_slopes.jpeg")
    plt.show()
def plot_localisation_data():
    file = "/Users/lukaslange/PycharmProjects/spacial_unmasking/data/results/added_result_files/localisation_accuracy_summed.csv"
    df_full = pd.read_csv(file)
    df_list = [df_full[df_full["stim_type"] == "pinknoise"], df_full[df_full["stim_type"] == "babble"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        df = df_list[i]
        # Calculate mean and standard deviation
        grouped_df = df.groupby("stim_dist").agg(
            mean_resp_dist=("resp_dist", "mean"),
            std_resp_dist=("resp_dist", "std")
        ).reset_index()

        # Plot the mean points
        sns.lineplot(data=grouped_df, x="stim_dist", y="mean_resp_dist",
                     marker="o", ax=ax)

        # Add error bars
        ax.errorbar(grouped_df["stim_dist"], grouped_df["mean_resp_dist"],
                    yerr=grouped_df["std_resp_dist"], fmt='o', color='blue', capsize=5)
        ax.set_title(f'stimulus type: {str(df["stim_type"].unique()[0])}')
        ax.set_xlabel('stimulus distance in m')
        ax.set_ylabel('estimated distance in m')
        # Add a black line with slope = 1 and intercept = 0
        x_vals = np.array(ax.get_xlim())  # Get current x-axis limits
        y_vals = x_vals  # Since slope = 1 and intercept = 0, y = x
        ax.plot(x_vals, y_vals, '--', color='black', label='y = x (slope = 1)')
        ax.set_aspect('equal', adjustable='box')


    plt.tight_layout()
    # plt.show()
    plt.savefig(DIR / "data" / "results" / "figs" / "fig_localisation_data.pdf")

def plot_numerosity_judgement_individual_data():
    file = "/Users/lukaslange/PycharmProjects/spacial_unmasking/data/results/added_result_files/numerosity_judgement_round_2_covariates_processed.csv"
    df = pd.read_csv(file)
    plt.figure(figsize=(6, 6))
    sns.lineplot(df, x="stim_number", y="resp_number", errorbar = None, hue="subject_id")
    # Add a black line with slope = 1 and intercept = 0
    x_vals = np.array(plt.gca().get_xlim())  # Get current x-axis limits
    y_vals = x_vals  # Since slope = 1 and intercept = 0, y = x
    plt.plot(x_vals, y_vals, '--', color='black')

    # Place the legend to the right of the plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Subjects")

    plt.gca().set_aspect('equal', adjustable='box')

    plt.xlabel("stimulus number")
    plt.ylabel("response number")

    plt.tight_layout()
    plt.savefig(DIR / "data" / "results" / "figs" / "fig_numerosity_judgement_individual_data.jpeg")

def plot_numerosity_judgement_data_per_stim_type():
    file = "/Users/lukaslange/PycharmProjects/spacial_unmasking/data/results/added_result_files/numerosity_judgement_round_2_covariates_processed.csv"
    df = pd.read_csv(file)
    df_full = pd.read_csv(file)
    df_list = [df_full[df_full["stim_type"] == "forward"], df_full[df_full["stim_type"] == "reversed"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        df = df_list[i]
        # Calculate mean and standard deviation
        grouped_df = df.groupby("stim_number").agg(
            mean_resp_number=("resp_number", "mean"),
            std_resp_number=("resp_number", "std")
        ).reset_index()

        # Plot the mean points
        sns.lineplot(data=grouped_df, x="stim_number", y="mean_resp_number",
                     marker="o", ax=ax)

        # Add error bars
        ax.errorbar(grouped_df["stim_number"], grouped_df["mean_resp_number"],
                    yerr=grouped_df["std_resp_number"], fmt='o', color='blue', capsize=5)
        ax.set_title(f'stimulus type: {str(df["stim_type"].unique()[0])}')
        ax.set_xlabel('stimulus number')
        ax.set_ylabel('response number')

        # Add a black line with slope = 1 and intercept = 0
        x_vals = np.array(ax.get_xlim())  # Get current x-axis limits
        y_vals = x_vals  # Since slope = 1 and intercept = 0, y = x
        ax.plot(x_vals, y_vals, '--', color='black', label='y = x (slope = 1)')
        ax.set_aspect('equal', adjustable='box')


    plt.tight_layout()
    # plt.show()
    plt.savefig(DIR / "data" / "results" / "figs" / "fig_numerosity_judgement_data_per_stim_type.pdf")

def plot_numerosity_judgement_data():

    file = "/Users/lukaslange/PycharmProjects/spacial_unmasking/data/results/added_result_files/numerosity_judgement_round_2_covariates_processed.csv"
    df = pd.read_csv(file)

    # Calculate mean and standard deviation
    grouped_df = df.groupby("stim_number").agg(
        mean_resp_number=("resp_number", "mean"),
        std_resp_number=("resp_number", "std")
    ).reset_index()

    # Plot the mean points
    sns.lineplot(data=grouped_df, x="stim_number", y="mean_resp_number",
                 marker="o")
    plt.errorbar(grouped_df["stim_number"], grouped_df["mean_resp_number"],
                    yerr=grouped_df["std_resp_number"], fmt='o', color='blue', capsize=5)

    # Add a black line with slope = 1 and intercept = 0
    x_vals = np.array(plt.xlim())  # Get current x-axis limits
    y_vals = x_vals  # Since slope = 1 and intercept = 0, y = x
    plt.plot(x_vals, y_vals, '--', color='black', label='y = x (slope = 1)')
    plt.gca().set_aspect('equal', adjustable='box')

    plt.xlabel('stimulus number')
    plt.ylabel('response number')

    for i in range(len(grouped_df)):
        x_value = grouped_df.loc[i, "stim_number"]
        mean_value = grouped_df.loc[i, "mean_resp_number"]
        sem_value = grouped_df.loc[i, "std_resp_number"]
        print(str(x_value) + ", " + str(mean_value) + ", " + str(sem_value))


    plt.tight_layout()
    # plt.show()
    plt.savefig(DIR / "data" / "results" / "figs" / "fig_numerosity_judgement_data.pdf")

def plot_spatial_unmasking_slopes():
    file = "/Users/lukaslange/PycharmProjects/spacial_unmasking/data/results/added_result_files/spacial_unmasking_summed_and_processed.csv"
    df = pd.read_csv(file)

    """sns.scatterplot(data=df, x="abs_spatial_separation", y="relative_threshold", hue="subject")"""
    # Fit and plot a regression line for each subject
    subjects = df['subject'].unique()

    for subject in subjects:
        subject_data = df[df['subject'] == subject]
        X = subject_data['abs_spatial_separation'].values
        y = subject_data['relative_threshold'].values

        # Fit linear regression using numpy's polyfit (degree 1 for linear)
        slope, intercept = np.polyfit(X, y, 1)
        print("slope = " + str(slope))
        print("intercept = " + str(intercept))
        # Calculate the regression line only within the range of the data
        x_vals = np.linspace(X.min(), X.max(), 100)
        y_vals = slope * x_vals + intercept

        # Plot regression line without adding to the legend
        plt.plot(x_vals, y_vals, linestyle='-', linewidth=2, color='gray', alpha=0.7)

        # Plot regression line
        plt.plot(x_vals, y_vals, label=f'Regression: Subject {subject}', linestyle='-', linewidth=2)
    # Add a black line with slope = 1 and intercept = 0

    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Subjects")

    plt.xlabel("absolute spatial separation in m")
    plt.ylabel("relative threshold in dB")

    plt.tight_layout()
    # plt.savefig(DIR / "data" / "results" / "figs" / "fig_localisation_individual_data_slopes.jpeg")
    plt.savefig(DIR / "data" / "results" / "figs" / "fig_spatial_unmasking_slopes.pdf")
    plt.show()


def get_la_rmse_variance():
    file = "/Users/lukaslange/PycharmProjects/spacial_unmasking/data/results/added_result_files/numerosity_judgement_processed.csv"
    df = pd.read_csv(file)
    rmse_list = list()
    subjects = df['subject_id'].unique()

    for subject in subjects:
        subject_data = df[df['subject_id'] == subject]
        rmse = subject_data["la_rmse"].values[0]
        rmse_list.append(rmse)

    variance = np.var(rmse_list, ddof=1)
    standard_deviation = np.std(rmse_list)
    mean = np.mean(rmse_list)
    coefficient_of_variation = standard_deviation / mean
    print(standard_deviation)
    print(variance)
    print(mean)
    print(coefficient_of_variation)
    return variance

def get_new_su_parameters():
    file = "/Users/lukaslange/PycharmProjects/spacial_unmasking/data/results/added_result_files/numerosity_judgement_processed.csv"
    df = pd.read_csv(file)
    df["su_parameter"] = (df["su_slope"] + df["su_slope_closest_speaker"]) / 2
    print(df["su_parameter"].unique())

def get_la_rmse_per_stimtype(subject_id):
    file = f"/Users/lukaslange/PycharmProjects/spacial_unmasking/data/results/results_localisation_accuracy_{subject_id}.csv"
    df = pd.read_csv(file)

    df_pinknoise = df[df["stim_type"] == "pinknoise"]
    df_babble = df[df["stim_type"] == "babble"]

    la_rmse_pinknoise = np.sqrt(np.mean(abs(df_pinknoise["stim_dist"] - df_pinknoise["resp_dist"])))
    la_rmse_babble = np.sqrt(np.mean(abs(df_babble["stim_dist"] - df_babble["resp_dist"])))

    return la_rmse_pinknoise, la_rmse_babble

def get_la_mae_per_stimtype(subject_id):
    file = f"/Users/lukaslange/PycharmProjects/spacial_unmasking/data/results/results_localisation_accuracy_{subject_id}.csv"
    df = pd.read_csv(file)

    df_pinknoise = df[df["stim_type"] == "pinknoise"]
    df_babble = df[df["stim_type"] == "babble"]

    la_mae_pinknoise = np.mean(abs(df_pinknoise["stim_dist"] - df_pinknoise["resp_dist"]))
    la_mae_babble = np.mean(abs(df_babble["stim_dist"] - df_babble["resp_dist"]))

    return la_mae_pinknoise, la_mae_babble

def get_r2_from_rsme(subject_id):
    file = f"/Users/lukaslange/PycharmProjects/spacial_unmasking/data/results/results_localisation_accuracy_{subject_id}.csv"
    df = pd.read_csv(file)
    # Observed and predicted values
    observed = df["resp_dist"]
    predicted = df["stim_dist"]

    # Calculate RMSE correctly
    rmse = np.sqrt(np.mean((observed - predicted) ** 2))

    # Number of observations
    n = len(observed)

    # Total Sum of Squares (TSS)
    tss = np.sum((observed - np.mean(observed)) ** 2)

    # Residual Sum of Squares (RSS) calculated directly from residuals
    rss = np.sum((observed - predicted) ** 2)

    # Calculate R-squared
    r_squared = 1 - (rss / tss)
    return r_squared


def plot_spectral_coverage_effects():
    # Prepare the data for plotting
    data = {
        'stim_number_f': ['2', '3', '4', '5', '6'],
        'estimate': [0.65915, 0.71827, 0.68672, 0.48267, 0.21441],
        'std_error': [0.23088, 0.16080, 0.15219, 0.19980, 0.27549],
        'p_value': [0.00432, 8.08e-06, 6.53e-06, 0.01573, 0.43643]
    }

    df = pd.DataFrame(data)

    # Determine significance levels for the asterisks
    df['significance'] = df['p_value'].apply(
        lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else '')

    # Plot the data
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x='stim_number_f', y='estimate', ci=None, palette="Blues", width=0.5)

    # Add error bars manually
    plt.errorbar(x=df['stim_number_f'], y=df['estimate'], yerr=df['std_error'], fmt='none', color='black', capsize=5)

    # Add significance asterisks
    for i, row in df.iterrows():
        plt.text(i, row['estimate'] + row['std_error'] + 0.05, row['significance'], ha='center', va='bottom',
                 color='black', fontsize=12)

    plt.ylim(-0.1,1)

    # Label the axes
    plt.xlabel('stimulus number')
    plt.ylabel('effect of spectral coverage')

    plt.savefig(DIR / "data" / "results" / "figs" / "fig_spectral_coverage_effect.pdf")
    plt.show()

def plot_la_rmse_effects():
    # Prepare the data for plotting
    data = {
        'stim_number_f': ['2', '3', '4', '5', '6'],
        'estimate': [0.65600, 0.50081, 0.34562, 0.19043, 0.03524]
    }

    df = pd.DataFrame(data)


    # Plot the data
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x='stim_number_f', y='estimate', ci=None, palette="Blues", width=0.5)


    plt.ylim(-0.1, 1)

    # Label the axes
    plt.xlabel('stimulus number')
    plt.ylabel('effect of la_rmse')

    plt.savefig(DIR / "data" / "results" / "figs" / "fig_la_rmse_effect.pdf")
    plt.show()

def plot_stim_type_effects():
    # Prepare the data for plotting
    data = {
        'stim_number_f': ['2', '3', '4', '5', '6'],
        'estimate': [0.29106, 0.25461, 0.21816, 0.18171, 0.14526]
    }

    df = pd.DataFrame(data)


    # Plot the data
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x='stim_number_f', y='estimate', ci=None, palette="Blues", width=0.5)


    plt.ylim(0, 0.3)

    # Label the axes
    plt.xlabel('stimulus number')
    plt.ylabel('effect of stimulus type (reversed)')

    plt.savefig(DIR / "data" / "results" / "figs" / "fig_la_rmse_effect.pdf")
    plt.show()

def plot_la_power_function():
    file= "/Users/lukaslange/PycharmProjects/spacial_unmasking/data/results/added_result_files/localisation_accuracy_summed.csv"
    df = pd.read_csv(file)
    # df = df[df["stim_type"] == "pinknoise"]

    grouped_df = df.groupby("stim_dist").agg(
        mean_resp_dist=("resp_dist", "mean"),
        std_resp_dist=("resp_dist", "std")
    ).reset_index()

    log_distance = np.log(df["stim_dist"])
    log_response = np.log(df["resp_dist"])

    # Perform linear regression on the transformed data
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(log_distance, log_response)

    # Extract parameters
    a = slope
    k = np.exp(intercept)

    print(f"Estimated a: {a}")
    print(f"Estimated k: {k}")
    print(f"Estimated r: {r_value}")
    print(f"Estimated p: {p_value}")
    print(f"Estimated intercept: {intercept}")

    # Generate fitted response values using the power function
    fitted_response = k * df["stim_dist"] ** a

    # Plotting the data and the fitted line
    plt.figure(figsize=(8, 6))

    # Scatter plot of original data points
    plt.scatter(grouped_df["stim_dist"], grouped_df["mean_resp_dist"], color='blue', label='Mean Participant Responses')

    # Plot the fitted power function line
    plt.plot(df["stim_dist"], fitted_response, color='red', label=f'Fitted Line: response = {k:.2f} * distance^{a:.2f}')


    # Set both axes to logarithmic scale
    plt.xscale('log')
    plt.yscale('log')

    # Set limits for x and y axes
    plt.xlim(1.5, 15)  # adjust according to your data range
    plt.ylim(1.5, 15)  # adjust according to your data range

    plt.gca().set_aspect('equal', adjustable='box')

    # Labels and title
    plt.xlabel('Stimulus Distance in m (log scale)')
    plt.ylabel('Response Distance in m (log scale)')
    plt.legend()

    # Show grid for better readability on log scale
    plt.grid(True, which="both", ls="--", linewidth=0.5)

    plt.savefig(DIR / "data" / "results" / "figs" / "fig_la_power_function_fit.pdf")
    # Display the plot
    plt.show()


def plot_la_and_sc_effects():
    data_sc = {
        'effect': ['sc', 'sc', 'sc', 'sc', 'sc'],
        'stim_number_f': ['2', '3', '4', '5', '6'],
        'estimate': [0.63682, 0.73396, 0.76158 , 0.41841, 0.22277]
    }

    data_la = {
        'effect': ['la', 'la', 'la', 'la', 'la'],
        'stim_number_f': ['2', '3', '4', '5', '6'],
        'estimate': [-0.61952, -0.45789, -0.33765, 0.14106, 0.23374]
    }

    df_sc = pd.DataFrame(data_sc)
    df_la = pd.DataFrame(data_la)
    df = pd.concat([df_sc, df_la])

    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x='stim_number_f', y='estimate', ci=None, hue='effect', width=0.5)

    plt.ylim(-1, 1)

    # Label the axes
    plt.xlabel('stimulus number')
    plt.ylabel('effect')
    plt.savefig(DIR / "data" / "results" / "figs" / "fig_la_r2_sc_effects.pdf")
    plt.show()

if __name__ == "__main__":
    process_nj_data()