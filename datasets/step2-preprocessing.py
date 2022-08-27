#!/usr/bin/env python3
# --------------------------------------------------------------
# This program run the preprocessing steps of CIC-IDS2017.
#
# Author: Mahendra Data - mahendra.data@ub.ac.id
# License: BSD 3 clause
# --------------------------------------------------------------

import pandas as pd
import numpy as np

from sys import argv
from os import path

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

# --- PREPARATION ---

label_col = "Label"  # A column which contains dataset labels
dataset_dir = "CIC-IDS2017-MachineLearning"  # A directory which contains the extracted CIC-IDS2017 dataset

print("--- Map of simplification version of the labels ---")
# Simplification version of the labels
label_map = {
    "Label": {
        "BENIGN": 0,
        "FTP-Patator": 1,
        "SSH-Patator": 2,
        "DoS Hulk": 3,
        "DoS GoldenEye": 4,
        "DoS slowloris": 5,
        "DoS Slowhttptest": 6,
        "Heartbleed": 7,
        "Web Attack � Brute Force": 8,
        "Web Attack � XSS": 9,
        "Web Attack � Sql Injection": 10,
        "Infiltration": 11,
        "Bot": 12,
        "PortScan": 13,
        "DDoS": 14
    }
}
for k, v in label_map[label_col].items():
    print("{:>2} - {}".format(v, k))

# Dataset file names
file_names = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
]
file_names = [path.join(dataset_dir, f) for f in file_names]  # Path of dataset files


# A function to change the labels in the dataset
def relabeling(df_target: pd.DataFrame):
    global label_col
    global label_map

    print("\nOriginal label:")
    print(df_target[label_col].value_counts())

    # Replace label
    df_target.replace(label_map, inplace=True)

    print("\nConverted label:")
    print(df_target[label_col].value_counts())


# Run this preprocessing steps for all dataset files
for f in file_names:
    print("\n--- {} ---".format(f))
    df = pd.read_csv(f, skipinitialspace=True)

    # --- STEP 1: Relabeling ---
    relabeling(df)

    # --- STEP 2: Remove duplicate column ---
    df = df.drop(columns=['Fwd Header Length.1'])

    # --- STEP 3: Remove rows contain inf or NaN values ---
    # Replace inf values with NaN.
    df.replace([np.inf], np.nan, inplace=True)
    print("Remove {} rows contain inf or NaN value.".format(df.isna().any(axis=1).sum()))
    df.dropna(inplace=True)

    # Save
    print("Saving dataset: {}".format(f))
    df.to_csv(f, index=False)
