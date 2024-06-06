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

def start_trial(sub_id, stim_type="pinknoise", n_reps=3):
    sounds_dict = util.get_sounds_dict(stim_type=stim_type)
    seq = slab.Trialsequence(conditions=list(range(11)), n_reps=3)
    """conditions = list(range(10)) * n_reps
    while True:
        np.random.shuffle(conditions)
        valid = True
        for i in range(1, len(conditions)):
            if conditions[i] == conditions[i - 1]:
                valid = False
                break
        if valid:
            break
    print(conditions)"""
    for trial in seq:
        event_id = seq.this_n
        filename, sound = util.get_sounds_with_filenames(sounds_dict=sounds_dict, n=1, randomize=True)
        filename = filename[0]
        sound = sound[0]
        print(trial)
        speaker = freefield.pick_speakers(trial)[0]
        print(speaker)
        util.set_multiple_signals(signals=[sound], speakers=[speaker], equalize=True)
        freefield.play(kind=1, proc="RX81")
        response = get_slider_value()
        print(response)
        save_results(event_id=event_id, sub_id=sub_id, stim_type=stim_type, response=response,
                     speaker_distance=speaker.distance, sound_filename=filename)

    return


def save_results(event_id, sub_id, stim_type, response, speaker_distance, sound_filename):
    file_name = DIR / "data" / "results" / f"results_localisation_accuracy_{sub_id}.csv"


    results = {"event_id": event_id,
               "sub_id": sub_id,
               "stim_type": stim_type,
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
            last_received = lines[-2].rstrip()
            if last_received:
                last_received = int(last_received)
                if in_metres:
                    last_received = np.interp(last_received, xp=[0, 1023], fp=[0, 15]) - 1.5
                return last_received


if __name__ == "__main__":
    plot_results(99)
