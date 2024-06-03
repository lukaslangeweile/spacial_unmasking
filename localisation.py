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
import serial

port = "COM5"
slider = serial.Serial(port, baudrate=9600, timeout=0, rtscts=False)
sounds = {}
DIR = pathlib.Path(os.curdir)


def initialize(sound_type="syllable"):
    procs = [["RX81", "RX8", DIR / "data" / "rcx" / "cathedral_play_buf.rcx"],
             ["RP2", "RP2", DIR / "data" / "rcx" / "button_numpad.rcx"]]
    freefield.initialize("cathedral", device=procs, zbus=False, connection="USB")
    freefield.SETUP = "cathedral"
    freefield.SPEAKERS = freefield.read_speaker_table()
    freefield.set_logger("DEBUG")
    if sound_type == "syllable":
        stim_DIR = DIR / "data" / "stim_files" / "tts-numbers_reversed"
    elif sound_type == "USO":
        stim_DIR = DIR / "data" / "stim_files" / "uso_300ms"
    sound_files = [file for file in stim_DIR.iterdir()]
    for file in sound_files:
        sounds.update({os.path.basename(file): slab.Sound.read(str(file))})


def start_trial(sub_id, sound_type="pinknoise", n_reps=30):
    for i in range(n_reps):
        event_id = i
        if sound_type == "pinknoise":
            sound = slab.Sound.pinknoise(duration=0.3)
            filename = "pinknoise"
        else:
            filename, sound = get_sounds_with_filenames(1)
            filename = filename[0]
            sound = sound[0]
        speaker = freefield.pick_speakers(np.random.randint(0, 11))[0]
        sound = apply_mgb_equalization(signal=sound, speaker=speaker)
        freefield.set_signal_and_speaker(signal=sound, speaker=speaker, equalize=False)
        freefield.play(kind=1, proc="RX81")
        response = get_slider_value()
        print(response)
        freefield.flush_buffers(processor="RX81")
        save_results(event_id=event_id, sub_id=sub_id, response=response,
                     speaker_distance=speaker.distance, sound_filename=filename)

    return

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

    df_curr_results = pd.DataFrame(results, index=[0])
    df_curr_results.to_csv(file_name, mode='a', header=not os.path.exists(file_name))


def get_slider_value(serial_port=slider, in_metres=True):
    serial_port.flushInput()
    buffer_string = ''
    while True:
        buffer_string = buffer_string + serial_port.read(serial_port.inWaiting()).decode("ascii")
        if '\n' in buffer_string:
            lines = buffer_string.split('\n')  # Guaranteed to have at least 2 entries
            last_received = int(lines[-2].rstrip())
            if in_metres:
                last_received = np.interp(last_received, xp=[0, 1023], fp=[0, 15])
            return last_received

