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
import logging

DIR = pathlib.Path(os.curdir)

logging.basicConfig(filename="localisation.log", level=logging.ERROR)

port = "COM5"
try:
    slider = serial.Serial(port, baudrate=9600, timeout=0, rtscts=False)
except serial.SerialException as e:
    logging.error(f"Failed to connect to the serial port {port}: {e}")

DIR = pathlib.Path(os.curdir)

def start_experiment(sub_id, block_id, stim_type="pinknoise", n_reps=10):
    try:
        trial_index = 0
        sounds_dict = util.get_sounds_dict(stim_type=stim_type)
        seq = slab.Trialsequence(conditions=list(range(11)), n_reps=n_reps)
        max_n_samples = util.get_max_n_samples(util.get_stim_dir(stim_type))

        logging.info("Beginning localisation experiment.")

        for trial in seq:
            logging.info(f"Presenting stimuli of stim_type = {stim_type} at speaker with index = {trial}")
            event_id = seq.this_n
            filename, sound = util.get_sounds_with_filenames(sounds_dict=sounds_dict, n=1, randomize=True)
            filename = filename[0]
            sound = sound[0]
            print(trial)
            speaker = freefield.pick_speakers(trial)[0]
            print(speaker)
            util.set_multiple_signals(signals=[sound], speakers=[speaker], equalize=True, max_n_samples=max_n_samples)
            freefield.play(kind=1, proc="RX81")
            util.start_timer()
            response = get_slider_value()
            reaction_time = util.get_elapsed_time()
            print(response)
            logging.info(f"Got response = {response}. Actual distance = {speaker.distance}")
            save_results(event_id=event_id, sub_id=sub_id, block_id=block_id, trial_index=trial_index, sound=sound, stim_type=stim_type, response=response,
                         speaker=speaker, sound_filename=filename, reaction_time=reaction_time)
            trial_index += 1
    except Exception as e:
        logging.error(f"An error occured in start_experiment: {e}")
        print(f"An error occurred: {e}")



def save_results(event_id, sub_id, trial_index, block_id, stim_type, sound, response, speaker, sound_filename, reaction_time):
    try:
        file_name = DIR / "data" / "results" / f"results_localisation_accuracy_{sub_id}.csv"


        results = {"event_id": None,
                   "timestamp": util.get_timestamp(),
                   "subject_id": sub_id,
                   "session_index": 3,
                   "plane": "distance",
                   "setup": "cathedral",
                   "task": "localisation_accuracy",
                   "block": block_id,
                   "trial_index": trial_index,
                   "stim_type": stim_type,
                   "headpose_offset_azi": None,
                   "headpose_offset_ele": None,
                   "stim_filename": os.path.basename(sound_filename),
                   "stim_level": sound.level, #TODO: mgb or level?
                   "speaker_id": speaker.index,
                   "speaker_proc": speaker.analog_proc,
                   "speaker_chan": speaker.analog_channel,
                   "stim_azi": speaker.azimuth,
                   "stim_ele": speaker.elevation,
                   "stim_dist": speaker.distance,
                   "resp_azi": 0,
                   "resp_ele": 0,
                   "resp_dist": response,
                   "reaction_time": reaction_time}

        df_curr_results = pd.DataFrame(results, index=[0])
        df_curr_results.to_csv(file_name, mode='a', header=not os.path.exists(file_name))
    except Exception as e:
        logging.error(f"An error occurred in save_results: {e}")
        print(f"An error occured: {e}")

def plot_results(sub_id):
    try:
        filepath = DIR / "data" / "results" / f"results_localisation_accuracy_{sub_id}.csv"
        df = pd.read_csv(filepath)
        sns.pointplot(df, x="speaker_distance", y="response", hue="stim_type")
        plt.xlabel("Speaker Distance in m")
        plt.ylabel("Estimated Distance in m")
        plt.title("Localisation Judgement")
        fig = plt.gcf()
        plt.show()
        plt.draw()
    except FileNotFoundError as e:
        logging.error(f"Result file not found: {e}")
        print(f"Result file not found: {e}")
    except Exception as e:
        logging.error(f"An error occurred in plot_results: {e}")
        print(f"An error occured: {e}")


def get_slider_value(serial_port=slider, in_metres=True):
    try:
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
    except serial.SerialException as e:
        logging.error(f"Serial communication error: {e}")
        print(f"Serial communication error: {e}")
    except ValueError as e:
        logging.error(f"Value conversion error: {e}")
        print(f"Value conversion error: {e}")
    except Exception as e:
        logging.error(f"An error occurred in get_slider_value: {e}")
        print(f"An error occured: {e}")

if __name__ == "__main__":
    plot_results(101)
