import freefield
import slab
import pickle
import os
import pathlib
import time
import numpy

DIR = pathlib.Path(os.curdir)
NORMALISATION_METHOD = None
STIM_TYPE = None
num_dict = {"one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six":  6,
            "seven": 7,
            "eight": 8,
            "nine": 9}

def initialize_setup(normalisation_method = "RMS"):
    procs = [["RX8", "RX8", DIR / "data" / "rcx" / "cathedral.rcx"],
             ["RP2", "RP2", DIR / "data" / "rcx" / "button.rcx"]]
    freefield.initialize("cathedral", device=procs, zbus=False, connection="USB")
    normalisation_file = DIR / "data" / "calibration" / f"calibration_pinknoise_{normalisation_method}.pkl"
    freefield.load_equalization(str(normalisation_file))

def start_trial(sub_id, masker_type, stim_type, normalisation_method):
    target_speaker = freefield.pick_speakers(5)[0]
    spacial_unmask_from_peripheral_speaker(start_speaker=0, target_speaker=target_speaker, sub_id=sub_id,
                                           masker_type=masker_type, stim_type=stim_type,
                                           normalisation_metod=normalisation_method)
    spacial_unmask_from_peripheral_speaker(start_speaker=10, target_speaker=target_speaker, sub_id=sub_id,
                                          masker_type=masker_type, stim_type=stim_type,
                                          normalisation_metod=normalisation_method)

def get_correct_response(file):
    for key in num_dict:
        if key in file:
            return num_dict.get(key)

def get_possible_files(sex=None, number=None, talker=None, exclude=False):
    possible_files = []
    stim_dir = DIR / "data" / "stim_files" / "tts"  #TODO: Add files and directory
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
            possible_files.append(file)
    else:
        for file in os.listdir(stim_dir):
            if sex is not None and sex in str(file):
                continue
            if number is not None and number in str(file):
                continue
            if talker is not None and talker in str(file):
                continue
            possible_files.append(file)

    return possible_files

def get_random_file(files):
    return numpy.random.choice(files)

def get_target_and_masker_file():
    stimuli = get_possible_files()
    target_file = get_random_file(stimuli)
    correct_response = get_correct_response(target_file)
    masker_file = get_random_file(get_possible_files(number=correct_response, exclude=True))
    return target_file, masker_file

def spacial_unmask_from_peripheral_speaker(start_speaker, target_speaker, sub_id, masker_type, stim_type, normalisation_method):
    if start_speaker > 5:
        iterator = list(range(10, 5, -1))
    else:
        iterator = list(range(5))

    for i in iterator:
        masking_speaker = freefield.pick_speakers(i)[0]
        stairs = slab.Staircase(start_val=(-20), n_reversals=5, step_sizes=[5, 3, 1])

        for level in stairs:
            target_file, masker_file = get_target_and_masker_file()
            target = slab.Sound.read(target_file)
            masker = slab.Sound.read(masker_file)
            freefield.apply_equalization(signal=target, speaker=target_speaker, level=True, frequency=False)
            freefield.apply_equalization(signal=masker, speaker=masking_speaker, level=True, frequency=False)
            target.level += level  # TODO: think about which level needs to be adjusted
            freefield.set_signal_and_speaker(signal=target, speaker=target_speaker, equalize=False)
            freefield.set_signal_and_speaker(signal=masker, speaker=masking_speaker, equalize=False)
            freefield.play(kind=1)
            while not freefield.read("response", "RP2"):
                time.sleep(0.1)
            response = freefield.read("response", "RP2")

            if response == get_correct_response(target_file):
                stairs.add_response(1)
            else:
                stairs.add_response(0)

            save_results(sub_id=sub_id, threshold=stairs.threshold(), distance_masker=masking_speaker.distance,
                         distance_target=target_speaker.distance, level_masker=masker.level, level_target=target.level,
                         masker_type=masker_type, stim_type=stim_type, normalisation_method=normalisation_method)

def save_results(sub_id, threshold, distance_masker, distance_target,
                 level_masker, level_target, masker_type, stim_type, normalisation_method):
    file_name = DIR / "data" / "results" / f"results_{sub_id}.pkl"
    results = {f"subject: {sub_id}": {"threshold": threshold,
                                      "distance_masker": distance_masker,
                                      "distance_target": distance_target,
                                      "level_masker": level_masker, "level_target": level_target,
                                      "masker_type": masker_type, "stim_type": stim_type,
                                      "normalisation_method": normalisation_method}}
    with open(file_name, 'wb') as f:  # save the newly recorded results
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)