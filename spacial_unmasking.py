import freefield
import slab
import pickle
import matplotlib
import logging
import os
import pathlib

DIR = pathlib.Path(os.curdir)
NORMALISATION_METHOD = None

def initialize_setup(normalisation_method = "default"):
    procs = [["RX8", "RX8", DIR / "data" / "rcx" / "cathedral.rcx"],
             ["RP2", "RP2", DIR / "data" / "rcx" / "button.rcx"]]
    freefield.initialize("cathedral", device=procs, zbus=False, connection="USB")
    NORMALISATION_METHOD = normalisation_method
    normalisation_file = DIR / "data" / "calibration" / f"calibration_pinknoise_{normalisation_method}.pkl"
    freefield.load_equalization(normalisation_method)


def start_trial(sub_id, masker_type, stim_type):

    # save results
    file_name = DIR / "data" / "results" / f"{sub_id}.pkl"
    results = {f"subject: {sub_id}": {"distance_masker": 0, "distance_target": 0, "level_masker": 0, "level_target": 0,
               "masker_type": 0, "stim_type": stim_type}, "normalisation_method": NORMALISATION_METHOD}
    with open(file_name, 'wb') as f:  # save the newly recorded calibration
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
