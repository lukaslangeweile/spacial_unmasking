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

DIR = DIR = pathlib.Path(os.curdir)

num_dict = {"one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six":  6,
            "seven": 7,
            "eight": 8,
            "nine": 9}

start_time = None

def initialize_setup():
    #initialize for the whole setup, so all 3 experiments can run
    procs = [["RX81", "RX8", DIR / "data" / "rcx" / "cathedral_play_buf.rcx"],
             ["RP2", "RP2", DIR / "data" / "rcx" / "button_numpad.rcx"]]
    freefield.initialize("cathedral", device=procs, zbus=False, connection="USB")
    freefield.SETUP = "cathedral"
    freefield.SPEAKERS = freefield.read_speaker_table()
    freefield.set_logger("DEBUG")
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

def get_speaker_normalisation_level(speaker, mgb_loudness=30):
    a, b, c = get_log_parameters(speaker.distance)
    return logarithmic_func(x=mgb_loudness, a=a, b=b, c=c)

def get_stim_dir(stim_type):
    if stim_type == "babble":
        stim_dir =  DIR / "data" / "stim_files" / "babble-numbers-reversed-n13-shifted_resamp_48828_resamp_24414"
    elif stim_type == "pinknoise":
        stim_dir = DIR / "data" / "stim_files" / "pinknoise_resamp_24414"
    elif stim_type == "countries" or stim_type == "countries_forward":
        stim_dir = DIR / "data" / "stim_files" / "tts-countries_n13_resamp_24414"
    elif stim_type == "countries_reversed":
        stim_dir = DIR / "data" / "stim_files" / "tts-countries_reversed_n13_resamp_24414"
    elif stim_type == "syllable":
        stim_dir = DIR / "data" / "stim_files" / "tts-numbers_n13_resamp_24414"
    elif stim_type == "uso":
        stim_dir = DIR / "data" / "stim_files" / "uso_300ms_resamp_24414"
    else:
        return
    return pathlib.Path(stim_dir)

def apply_mgb_equalization(signal, speaker, mgb_loudness=30, fluc=0):
    a, b, c = get_log_parameters(speaker.distance)
    signal.level = logarithmic_func(mgb_loudness + fluc, a, b, c)
    return signal

def record_stimuli_and_create_csv(stim_type="countries_forward"):
    sounds_list, filenames_list = get_sounds_with_filenames(n="all", stim_type="stim_type")
    for i in range(len(sounds_list)):
        for speaker in freefield.SPEAKERS:
            sound = sounds_list[i]
            sound = apply_mgb_equalization(signal=sound, speaker=speaker, mgb_loudness=30)
            rec_sound = freefield.play_and_record(speaker=speaker, sound=sound, compensate_delay=True,
                                                  equalize=False)
            sound_name = os.path.splitext(filenames_list[i])[0]
            filepath = DIR / "data" / "recordings" / " countries_forward" / f"sound-{sound_name}_distance-{speaker.distance}.wav"
            filepath = pathlib.Path(filepath)
            rec_sound.write(filename=filepath, normalise=False)
            time.sleep(2.7)

    #TODO: saving sound data to csv-file

    return

def get_sounds_dict(stim_type="babble"):
    stim_dir = get_stim_dir(stim_type)
    sounds_dict = {}

    for file in stim_dir.iterdir():
        sounds_dict.update({str(file): slab.Sound.read(file)})

    return sounds_dict

def get_sounds_with_filenames(sounds_dict, n="all", randomize=False):
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

def get_n_random_speakers(n):
    if n > len(freefield.SPEAKERS):
        print(len(freefield.SPEAKERS))
        print(n)
        raise ValueError("n cannot be greater than the length of the input list")
    random_indices = np.random.choice(len(freefield.SPEAKERS), n, replace=False)
    speakers = [freefield.pick_speakers(i)[0] for i in random_indices]
    return speakers, random_indices

def get_correct_response_number(file):
    for key in num_dict:
        if key in str(file):
            return num_dict.get(key)
def create_localisation_config_file():
    config_dir = DIR / "config"
    filename = config_dir / "localisation_config.json"
    config_data = {"sounds_dict_pinknoise": get_sounds_dict("pinknoise"),
                   "sounds_dict_syllable": get_sounds_dict("syllable"),
                   "sounds_dict_babble": get_sounds_dict("babble")}
    with open(filename, 'w') as config_file:
        json.dump(config_data, config_file, indent=4)

def create_numerosity_judgement_config_file():
    config_dir = DIR / "config"
    filename = config_dir / "localisation_config.json"
    config_data = {"num_dict": {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six":  6, "seven": 7,
                    "eight": 8, "nine": 9},
                   "talkers": ["p229", "p245", "p248", "p256", "p268", "p284", "p307", "p318"],
                   "n_sounds": [2, 3, 4, 5, 6]
    }

def read_config_file(experiment):
    if experiment == "localisation":
        filename = DIR / "config" / "localisation_config.json"
    elif experiment == "spacial_unmasking":
        filename = DIR / "config" / "spacial-unmasking_config.json"
    elif experiment == "numerosity_judgement":
        filename = DIR / "config" / "numerosity-judgement_config.json"
    else:
        return
    with open(filename, 'r') as config_file:
        config_data = json.load(config_file)
    return config_data

def set_multiple_signals(signals, speakers, equalize=True, mgb_loudness=30, fluc=0, max_n_samples=80000):
    for i in range(len(signals)):
        if equalize:
            signals[i] = apply_mgb_equalization(signals[i], speakers[i], mgb_loudness, fluc)
        speaker_index = speakers[i].index + 1
        data = np.pad(signals[i].data, ((0, max_n_samples - len(signals[i].data)), (0, 0)), 'constant')
        if len(data.shape) == 2 and data.shape[1] == 2:
            data = np.mean(data, axis=1)
        freefield.write(tag=f"data{i}", value=data, processors="RX81")
        freefield.write(tag=f"chan{i}", value=speaker_index, processors="RX81")
    time.sleep(0.2)
    for i in range(len(signals), 8):
        freefield.write(tag=f"chan{i}", value=99, processors="RX81")
        time.sleep(0.1)

def test_speakers():
    sound = slab.Sound.pinknoise(0.50)
    for speaker in freefield.SPEAKERS:
        set_multiple_signals([sound], [speaker], equalize=True)
        time.sleep(0.1)
        freefield.play(kind=1, proc="RX81")
        time.sleep(1.0)


def start_timer():
    global start_time
    start_time = time.time()


def get_elapsed_time(reset=True):
    global start_time
    if start_time is None:
        raise ValueError("Timer has not been started.")

    elapsed_time = time.time() - start_time

    if reset:
        start_time = None

    return elapsed_time

def get_timestamp():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def parse_country_or_number_filename(filepath):
    # Define the pattern to match the filename structure
    pattern = r"talker-(?P<talker>.+?)_sex-(?P<sex>[.\w]+)_text-(?P<text>[.\w]+)\.wav"
    filename = str(os.path.basename(filepath))
    match = re.match(pattern, filename)
    if match:
        talker = match.group("talker")
        sex = match.group("sex")
        text = match.group("text")
        return talker, sex, text
    else:
        print(f"Filename {filename} does not match the filepattern")
        return None

def create_resampled_stim_dirs(samplerate=24414):
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

def initialize_stim_recording():
    freefield.set_logger("DEBUG")
    freefield.SETUP = "cathedral"
    freefield.initialize(setup="cathedral", default="play_birec", connection="USB", zbus=False)
    freefield.SETUP = "cathedral"
    freefield.SPEAKERS = freefield.read_speaker_table()

def get_max_n_samples(stim_dirs):
    sounds_list = []


    if not isinstance(stim_dirs, list):
        stim_dirs = [stim_dirs]

    for directory in stim_dirs:
        directory_path = pathlib.Path(directory)
        if directory_path.is_dir():
            for file in directory_path.iterdir():
                if file.is_file() and file.suffix in ['.wav']:
                    try:
                        sounds_list.append(slab.Sound.read(file))
                    except Exception as e:
                        print(f"Error reading file {file}: {e}")
        else:
            print(f"{directory} is not a directory")

    max_n_samples = max(sound.n_samples for sound in sounds_list)
    return max_n_samples

def record_stimuli(stim_type, mgb_loudness=30):
    stim_dir = get_stim_dir(stim_type)
    recording_dir = DIR / "data" / "recordings" / stim_type
    if not os.path.exists(recording_dir):
        os.makedirs(recording_dir)
    for file in stim_dir.iterdir():
        sound = slab.Sound.read(file)
        basename = str(os.path.splitext(os.path.basename(file))[0])
        for speaker in freefield.SPEAKERS:
            sound = apply_mgb_equalization(sound, speaker, 30, 0)
            recording = freefield.play_and_record(speaker=speaker, sound=sound, equalize=False)

            recording_filename = f"sound-{basename}_mgb-level-{mgb_loudness}_distance-{speaker.distance}.wav"
            recording_filepath = recording_dir / recording_filename
            recording.write(recording_filepath)

def reverse_stimuli(stim_type):
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
if __name__ == "__main__":

    reverse_stimuli("countries")