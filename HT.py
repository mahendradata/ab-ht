#!/usr/bin/env python3
# --------------------------------------------------------------
# Author: Mahendra Data - mahendra.data@ub.ac.id
# License: BSD 3 clause
#
# Example: python HT.py conf/HT.json outputs/HT.log
# --------------------------------------------------------------

import argparse
import pprint

from river import tree

from lib.configuration import Configuration
from lib.util import Log
from lib.model.incremental import evaluate_model
from lib.dataset import IncrementalData


# Parse arguments
parser = argparse.ArgumentParser(description="CIC-IDS2017 classification using HoeffdingTree.")
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

# Model
model = (
    tree.HoeffdingTreeClassifier(
        split_criterion=conf.split_criterion,
        split_confidence=conf.split_confidence,
        grace_period=conf.grace_period
    )
)

samples = IncrementalData()  # Data management
evaluate_model("HT", model, conf.datasets, samples)  # Evaluate model

Log.write(Log.timestamp("STOP"))
