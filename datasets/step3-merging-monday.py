#!/usr/bin/env python3
# --------------------------------------------------------------
# This program run the preprocessing steps of CIC-IDS2017.
#
# Author: Mahendra Data - mahendra.data@ub.ac.id
# License: BSD 3 clause
# --------------------------------------------------------------

import pandas as pd

from sys import argv
from os import path

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

# --- PREPARATION ---

label_col = "Label"  # A column which contains dataset labels
dataset_dir = "CIC-IDS2017-MachineLearning"  # A directory which contains the extracted CIC-IDS2017 dataset

# Dataset file names
file_names = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv"
]
file_names = [path.join(dataset_dir, f) for f in file_names]  # Path of dataset files

# Read
df = [pd.read_csv(f, skipinitialspace=True) for f in file_names]
df = pd.concat(df, ignore_index=True)
# Save
file_name = path.join(dataset_dir, "Monday-Tuesday-WorkingHours.pcap_ISCX.csv")
print("Saving dataset: {}".format(file_name))
df.to_csv(file_name, index=False)
