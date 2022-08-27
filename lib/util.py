#!/usr/bin/env python3
# --------------------------------------------------------------
# Author: Mahendra Data - mahendra.data@ub.ac.id
# License: BSD 3 clause
# --------------------------------------------------------------

import os
import csv
import time
import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.metrics import classification_report


def counter(x: np.ndarray) -> dict:
    unique, counts = np.unique(x, return_counts=True)
    return dict(zip(unique, counts))


def shuffle(x: np.ndarray, y: np.ndarray):
    assert len(x) == len(y)
    p = np.random.permutation(len(x))
    return np.copy(x[p]), np.copy(y[p])


def report(y: np.ndarray, y_pred: np.ndarray):
    report_str = classification_report(y, y_pred)
    report_dict = classification_report(y, y_pred, output_dict=True)
    return report_str, report_dict


def confusion_matrix(y: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    y_pred_labels = np.unique(y_pred)
    y_labels = np.unique(y)

    matrix = pd.DataFrame(np.zeros((len(y_pred_labels), len(y_labels))),
                          index=y_pred_labels, columns=y_labels, dtype=int)

    for c in y_labels:
        c_pred = np.where(y_pred == c)[0]
        for p, num in counter(y[c_pred]).items():
            matrix.loc[c, p] = num

    return matrix


class Timer(object):
    def __init__(self):
        self.begin = time.time()

    def start(self):
        self.begin = time.time()

    def stop(self) -> float:
        return time.time() - self.begin


class Log(object):
    """Logger Class

    Log class has two main purposes:
    1. Print to the standard output and write to the designated output file.
    2. Convert and write the experimental data from dictionary to csv format.

    Examples
    --------

    >>> from eil.util import Log

    >>> Log.write("Hello World", 123)
    Hello World
    123

    >>> Log.write(Log.timestamp("Hello World"), 123)
    [16:12:19] Hello World
    123

    """

    BASE_NAME = "output"
    CONSOLE_OUTPUT = "{}.log".format(BASE_NAME)
    DATA_OUTPUT = "{}.data.csv".format(BASE_NAME)

    @staticmethod
    def set_output_path(path: str):
        Log.CONSOLE_OUTPUT = path
        Log.BASE_NAME = os.path.splitext(path)[0]
        Log.DATA_OUTPUT = "{}.data.csv".format(Log.BASE_NAME)

    @staticmethod
    def clear():
        f = open(Log.CONSOLE_OUTPUT, "w")
        f.close()

    @staticmethod
    def timestamp(message):
        return "[{}] {}".format(datetime.now().strftime("%H:%M:%S"), message)

    @staticmethod
    def write(*messages):
        with open(Log.CONSOLE_OUTPUT, "a") as f:
            for m in messages:
                print(m)
                f.write("{}\n".format(m))

    @staticmethod
    def write_exp(name: str, dtype: str, report_dict: dict):
        # Create header (if needed)
        if not os.path.isfile(Log.DATA_OUTPUT):
            with open(Log.DATA_OUTPUT, mode='w') as f:
                f_csv = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                f_csv.writerow(["name", "type", "class", "precision", "recall", "f1-score", "support"])

        # Write experimental data
        with open(Log.DATA_OUTPUT, mode='a') as f:
            f_csv = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for k, v in report_dict.items():
                if k == 'accuracy':
                    f_csv.writerow([name, dtype, k, "", "", v, ""])
                else:
                    f_csv.writerow([name, dtype, k, v["precision"], v["recall"], v["f1-score"], v["support"]])
