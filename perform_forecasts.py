import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import datetime as dt
import seaborn as sns
import argparse, sys, os, random
from data_loaders import load_dataset, VALID_REGIONS, EQTimeWindows

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

def rbfPredict(y, distMat, trainSize, eps):
    # Find the weights using only the training set
    trainMat = distMat[:trainSize,:]
    
    trainGram = np.exp(-trainMat**2 / eps)

    # Add bias parameters 
    trainGram = np.hstack([ np.ones((trainGram.shape[0], 1)), trainGram ])

    # Calculate weights
    w = np.linalg.inv(trainGram.transpose() @ trainGram) @ (trainGram.transpose() @ y[:trainSize])

    # Proceed to testing
    testMat = distMat[trainSize:,:]

    testGram = np.exp(-testMat**2 / eps)

    # To plot results on train data
    #plt.scatter(trainGram @ w, y[:trainSize])
    
    # Add bias parameters
    testGram = np.hstack([ np.ones([testGram.shape[0], 1]), testGram ])

    predicted = testGram @ w
    
    return predicted, y[trainSize:]

def getExperiments(distMat, all_predictor, all_expected, eps, trainSize, numIter=100, numBases=100):
    experiment = {
        "correlation": [],
        "predicted": []
    }

    for i in range(numIter):
        print(i, end=" ")

        selectIdx = np.arange(trainSize)
        np.random.shuffle(selectIdx)
        selectIdx = selectIdx[:numBases]

        #MIN = 10
        #MAX = 21
        #candidates = np.int32(np.round(np.cumsum(np.random.uniform(MIN, MAX, size=trainSize // MIN))))
        #candidates = candidates[candidates < trainSize]
        #np.random.shuffle(candidates)
        #selectIdx = candidates[:100] 

        selectDistances = distMat[:,selectIdx]

        predicted, real = rbfPredict(all_predictor, selectDistances, eps)

        mmags = np.array(all_expected)[trainSize:]

        # # Thresholds
        # a = 6
        # b = 3.2

        # tp = np.sum((mmags >  a) * (predicted >  b))
        # fp = np.sum((mmags <= a) * (predicted >  b))
        # fn = np.sum((mmags >  a) * (predicted <= b))
        # tn = np.sum((mmags <= a) * (predicted <= b))

        experiment["correlation"].append(np.corrcoef(predicted, mmags)[0,1])
        experiment["predicted"].append(predicted)
    
    return experiment

data = load_dataset(args.region, args.minmag)
eqtw = EQTimeWindows(data, args.inputw, args.outputw)
x_maxMag = eqtw.x_maxMag

fname = os.path.join(args.dir, f"{EXPERIMENT_NAME}.npy")
distanceMatrix = np.load(fname)

trainSize = distanceMatrix.shape[0] // 2
trainMat = distanceMatrix[:trainSize,:]
eps = 5 * np.mean(trainMat[trainMat > 0])**2

experiment = getExperiments(distanceMatrix, eqtw.y_logN, x_maxMag, eps, trainSize, numIter=20)

print(" ".join(experiment["correlation"]))