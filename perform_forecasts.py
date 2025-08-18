import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import datetime as dt
import seaborn as sns
import argparse, sys, os, random
from data_loaders import load_dataset, VALID_REGIONS

parser = argparse.ArgumentParser(prog='Calculates accuracy of prediction.')
parser.add_argument("--region",
        help="Region to analyze.",
        default="ja",
        choices=VALID_REGIONS)
parser.add_argument("--minmag",
        help="Minimum magnitude.",
        type=float,
        default=0)
parser.add_argument("--inputw",
        help="Size of the time window used for making predictions.",
        type=int,
        default=7)
parser.add_argument("--outputw",
        help="Size of the time window for which a prediction is sought.",
        type=int,
        default=1)
parser.add_argument("--tlambda",
        help="Multiplier of the lambda for timestamps.",
        type=float,
        default=100.0)
parser.add_argument("--limit-windows",
        help="Limits the number of earthquakes in each window to the X-percentile of the window sizes distribution.",
        action='store_true')
parser.add_argument("--limit-windows-p",
        help="Percentile to which window sizes should be truncated to.",
        type=float,
        default=0.8)
parser.add_argument("--dir",
        help="Read distance matrix from this directory.",
        type=str,
        default="./")
args = parser.parse_args()

print("Beginning program.")
print("Command executed: {}".format(' '.join(sys.argv)))
for i, j in args._get_kwargs():
    print("{}: {}".format(i, j))
print("==================")

EXPERIMENT_NAME = [
    f"distance-matrix",
    f"{args.region}",
    f"minmag{args.minmag}",
    f"inputw{args.inputw}",
    f"outputw{args.outputw}",
    f"tlambda{np.format_float_positional(args.tlambda, trim="-")}"
]
# if args.partial:
#     EXPERIMENT_NAME += [f"partial{args.partial_n}"]
EXPERIMENT_NAME = "-".join(EXPERIMENT_NAME)
print(f"Experiment name: {EXPERIMENT_NAME}")

def pkldump(obj, file):
    with open(file, "wb") as fp:
        pickle.dump(obj, fp)

def pklload(file):
    with open(file, "rb") as fp:
        return pickle.load(fp)

fname = os.path.join(args.dir, f"{EXPERIMENT_NAME}.npy")
distanceMatrix = np.load(fname)

print(distanceMatrix)