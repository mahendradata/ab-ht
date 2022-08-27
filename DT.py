#!/usr/bin/env python3
# --------------------------------------------------------------
# Author: Mahendra Data - mahendra.data@ub.ac.id
# License: BSD 3 clause
#
# Example: python DT.py conf/DT.json outputs/DT.log
# --------------------------------------------------------------

import argparse
import pprint

from sklearn.tree import DecisionTreeClassifier

from lib.configuration import Configuration
from lib.util import Log
from lib.model.batch import evaluate_model
from lib.dataset import BatchData


# Parse arguments
parser = argparse.ArgumentParser(description="CIC-IDS2017 classification using Decision Tree Classifier.")
parser.add_argument("--seed", nargs='?', help="Seed value.", type=int)
parser.add_argument("configuration", help="Path to program configuration.", type=str)
parser.add_argument("output", help="Path to output file.", type=str)
args = parser.parse_args()

# Log configuration
Log.set_output_path(args.output)  # Set output path
Log.clear()  # Clear output file
Log.write(Log.timestamp("START"))

# Configuration
conf = Configuration.read_json(args.configuration)
Log.write("Configuration:", pprint.pformat(conf))

model = DecisionTreeClassifier(criterion=conf.criterion, random_state=args.seed)  # Model

samples = BatchData()  # Data management
evaluate_model("DecisionTree", model, conf.datasets, samples, args.seed)  # Evaluate model

Log.write(Log.timestamp("STOP"))
