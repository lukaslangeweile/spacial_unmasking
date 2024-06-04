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
import util

port = "COM5"
slider = serial.Serial(port, baudrate=9600, timeout=0, rtscts=False)
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


def start_trial(sub_id, stim_type="pinknoise", n_reps=3):
    sounds_dict = util.get_sounds_dict(stim_type=stim_type)
    conditions = list(range(10)) * n_reps
    while True:
        np.random.shuffle(conditions)
        valid = True
        for i in range(1, len(conditions)):
            if conditions[i] == conditions[i - 1]:
                valid = False
                break
        if valid:
            break

    for i in range(n_reps * 11):
        event_id = i
        filename, sound = util.get_sounds_with_filenames(sounds_dict=sounds_dict, n=1, randomize=True)
        filename = filename[0]
        sound = sound[0]
        speaker = freefield.pick_speakers(conditions[i])[0]
        util.set_multiple_signals(signals=[sound], speakers=[speaker], equalize=True)
        freefield.play(kind=1, proc="RX81")
        response = get_slider_value()
        print(response)
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

def plot_results(sub_id):
    filepath = DIR / "data" / "results" / f"results_localisation_accuracy_{sub_id}.csv"
    df = pd.read_csv(filepath)
    sns.pointplot(df, x="speaker_distance", y="response", hue="stim_type")
    plt.xlabel("Speaker Distance in m")
    plt.ylabel("Estimated Distance in m")
    plt.title("Localisation Judgement")
    fig = plt.gcf()
    plt.show()
    plt.draw()


def get_slider_value(serial_port=slider, in_metres=True):
    serial_port.flushInput()
    buffer_string = ''
    while True:
        buffer_string = buffer_string + serial_port.read(serial_port.inWaiting()).decode("ascii")
        if '\n' in buffer_string:
            lines = buffer_string.split('\n')  # Guaranteed to have at least 2 entries
            last_received = int(lines[-2].rstrip())
            if in_metres:
                last_received = np.interp(last_received, xp=[0, 1023], fp=[0, 15]) - 1.5
            return last_received

if __name__ == "__main__":
    plot_results(99)
