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
        return quadratic2_func(x=input_level, a=a, b=b, c=c)
    except Exception as e:
        logging.error(f"An error occurred in get_speaker_normalisation_level: {e}")
        print(f"An error occurred: {e}")

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

if __name__ == "__main__":
    plot_task_order_effects(task="localisation_accuracy")


