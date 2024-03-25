import pickle
import seaborn as sns
import numpy as np
import pandas
import matplotlib.pyplot as plt
import slab
import freefield
import math
import pathlib
import os

DIR = pathlib.Path(os.getcwd())
data = DIR / "data" / "results" /"results_100.pkl"

with open(data, "rb") as f:
    loaded_results = pickle.load(f)

print(loaded_results)
sns_data = sns.load_dataset(loaded_results)

