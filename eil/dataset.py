# --------------------------------------------------------------
# Author: Mahendra Data - mahendra.data@ub.ac.id
# License: BSD 3 clause
# --------------------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split

# Data type of each column in CIC-IDS2017 dataset.
CICIDS2017_DTYPE = {
    "Destination Port": int,
    "Flow Duration": int,
    "Total Fwd Packets": int,
    "Total Backward Packets": int,
    "Total Length of Fwd Packets": int,
    "Total Length of Bwd Packets": int,
    "Fwd Packet Length Max": int,
    "Fwd Packet Length Min": int,
    "Fwd Packet Length Mean": float,
    "Fwd Packet Length Std": float,
    "Bwd Packet Length Max": int,
    "Bwd Packet Length Min": int,
    "Bwd Packet Length Mean": float,
    "Bwd Packet Length Std": float,
    "Flow Bytes/s": float,
    "Flow Packets/s": float,
    "Flow IAT Mean": float,
    "Flow IAT Std": float,
    "Flow IAT Max": int,
    "Flow IAT Min": int,
    "Fwd IAT Total": int,
    "Fwd IAT Mean": float,
    "Fwd IAT Std": float,
    "Fwd IAT Max": int,
    "Fwd IAT Min": int,
    "Bwd IAT Total": int,
    "Bwd IAT Mean": float,
    "Bwd IAT Std": float,
    "Bwd IAT Max": int,
    "Bwd IAT Min": int,
    "Fwd PSH Flags": int,
    "Bwd PSH Flags": int,
    "Fwd URG Flags": int,
    "Bwd URG Flags": int,
    "Fwd Header Length": int,
    "Bwd Header Length": int,
    "Fwd Packets/s": float,
    "Bwd Packets/s": float,
    "Min Packet Length": int,
    "Max Packet Length": int,
    "Packet Length Mean": float,
    "Packet Length Std": float,
    "Packet Length Variance": float,
    "FIN Flag Count": int,
    "SYN Flag Count": int,
    "RST Flag Count": int,
    "PSH Flag Count": int,
    "ACK Flag Count": int,
    "URG Flag Count": int,
    "CWE Flag Count": int,
    "ECE Flag Count": int,
    "Down/Up Ratio": int,
    "Average Packet Size": float,
    "Avg Fwd Segment Size": float,
    "Avg Bwd Segment Size": float,
    "Fwd Avg Bytes/Bulk": int,
    "Fwd Avg Packets/Bulk": int,
    "Fwd Avg Bulk Rate": int,
    "Bwd Avg Bytes/Bulk": int,
    "Bwd Avg Packets/Bulk": int,
    "Bwd Avg Bulk Rate": int,
    "Subflow Fwd Packets": int,
    "Subflow Fwd Bytes": int,
    "Subflow Bwd Packets": int,
    "Subflow Bwd Bytes": int,
    "Init_Win_bytes_forward": int,
    "Init_Win_bytes_backward": int,
    "act_data_pkt_fwd": int,
    "min_seg_size_forward": int,
    "Active Mean": float,
    "Active Std": float,
    "Active Max": int,
    "Active Min": int,
    "Idle Mean": float,
    "Idle Std": float,
    "Idle Max": int,
    "Idle Min": int,
    "Label": int
}


class IncrementalData(object):
    def __init__(self):
        # These variables are used to keep data from all batches
        self.eval_x = None
        self.eval_y = None

    @staticmethod
    def read_csv(f_path: str, col_label="Label", test_size=0.2, random_state=10):
        # Read dataset then split it into data and labels
        x = pd.read_csv(f_path, skipinitialspace=True, converters=CICIDS2017_DTYPE)  # data
        x = x.sample(frac=1)  # shuffle
        y = x.pop(col_label)  # labels

        # Split to train and test dataset
        train_x, eval_x, train_y, eval_y = train_test_split(x, y, stratify=y,
                                                            test_size=test_size, random_state=random_state)
        return train_x, eval_x, train_y, eval_y

    def read(self, f_path: str, col_label="Label", test_size=0.2, random_state=10):
        # Split to train and test dataset

        train_x, eval_x, train_y, eval_y = IncrementalData.read_csv(f_path, col_label, test_size, random_state)

        # Merge all evaluation data
        self.eval_x = pd.concat([self.eval_x, eval_x], ignore_index=False)
        self.eval_y = pd.concat([self.eval_y, eval_y], ignore_index=False)

        return train_x, self.eval_x, train_y, self.eval_y


class BatchData(object):
    def __init__(self):
        # These variables are used to keep data from all batches
        self.train_x = None
        self.train_y = None
        self.eval_x = None
        self.eval_y = None

    @staticmethod
    def read_csv(f_path: str, col_label="Label", test_size=0.2, random_state=10):
        # Read dataset then split it into data and labels
        x = pd.read_csv(f_path, skipinitialspace=True, converters=CICIDS2017_DTYPE)  # data
        x = x.sample(frac=1)  # shuffle
        y = x.pop(col_label)  # labels

        # Split to train and test dataset
        train_x, eval_x, train_y, eval_y = train_test_split(x, y, stratify=y,
                                                            test_size=test_size, random_state=random_state)
        return train_x, eval_x, train_y, eval_y

    def read(self, f_path: str, col_label="Label", test_size=0.2, random_state=10):
        # Split to train and test dataset

        train_x, eval_x, train_y, eval_y = BatchData.read_csv(f_path, col_label, test_size, random_state)

        # Merge all evaluation data
        self.train_x = pd.concat([self.train_x, train_x], ignore_index=False)
        self.train_y = pd.concat([self.train_y, train_y], ignore_index=False)
        self.eval_x = pd.concat([self.eval_x, eval_x], ignore_index=False)
        self.eval_y = pd.concat([self.eval_y, eval_y], ignore_index=False)

        return self.train_x.to_numpy(), self.eval_x.to_numpy(), self.train_y.to_numpy(), self.eval_y.to_numpy()
