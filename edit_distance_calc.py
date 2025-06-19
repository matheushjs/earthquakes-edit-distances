import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
import pickle
import datetime as dt
import copy
import seaborn as sns
import multiprocessing as mp
import time
import argparse, sys, os
from data_loaders import load_dataset

# import os, sys
# from astropy.time import Time
# from astropy.coordinates import solar_system_ephemeris, EarthLocation
# from astropy.coordinates import get_body_barycentric, get_body, get_moon, get_sun

def pkldump(obj, file):
    with open(file, "wb") as fp:
        pickle.dump(obj, fp)

def pklload(file):
    with open(file, "rb") as fp:
        return pickle.load(fp)

parser = argparse.ArgumentParser(prog='Calculator of Edit Distances')
parser.add_argument("--region",
        help="Region to analyze.",
        default="ja",
        choices=["ja", "gr", "nz", "jma"])
parser.add_argument("--minmag",
        help="Minimum magnitude.",
        type=float,
        default=0)
parser.add_argument("--nthreads",
        help="Number of threads to use.",
        type=int,
        default=8)
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
parser.add_argument("--outdir",
        help="Directory to output the results.",
        type=str,
        default="./")
args = parser.parse_args()

if args.minmag < 0 and args.region == "jma":
    args.minmag = 2.7

print("Beginning program.")
print("Command executed: {}".format(' '.join(sys.argv)))
for i, j in args._get_kwargs():
    print("{}: {}".format(i, j))
print("==================")

EXPERIMENT_NAME = "-".join([
    f"distance-matrix",
    f"{args.region}",
    f"minmag{args.minmag}",
    f"inputw{args.inputw}",
    f"outputw{args.outputw}",
    f"tlambda{args.tlambda:.3g}"
])
print(f"Experiment name: {EXPERIMENT_NAME}")

data = load_dataset(args.region, args.minmag)

ss = pd.read_csv("../sunspots.csv")

ss = ss[ss["year"] >= 2000].copy()
initDate = dt.datetime(2000, 1, 1)
newCol = []

for i in range(len(ss)):
    datum = ss.iloc[i,:]
    dateObj = dt.datetime(datum["year"], datum["month"], datum["day"])
    ndays = (dateObj - initDate).days
    
    newCol.append(ndays)

ss["day.number"] = newCol

allX_quakes = []
allX_maxMag = []
allX_meanMag = []
allX_logN = []
allX_sunspots = []

allX_quakes_N = []
allX_quakes_logN = []
#allX_seismicity = []

allY_dayNumbers = []

# Window size
W = args.inputw

# Prediction window
PRED_WINDOW = args.outputw

dayNumbers = data["day.number"] # To make things faster, save it in a variable
ssDayNumbers = ss["day.number"]

for i in range(W-1, max(dayNumbers) + 1 - PRED_WINDOW):
    quakeWindow = data[ (dayNumbers > i - W) * (dayNumbers <= i) ]
    predWindow  = data[ (dayNumbers > i) * (dayNumbers <= i+PRED_WINDOW) ]
    #predWindow  = nankaiData[ (nankaiData["day.number"] > i) * (nankaiData["day.number"] <= i+PRED_WINDOW) ]
    allY_dayNumbers.append(i + 1)
    
    ssWindow    = ss[ (ssDayNumbers > i - W) * (ssDayNumbers <= i) ]
    allX_sunspots.append(list(ssWindow["ssn (count)"]))
    
    if len(quakeWindow) > 0:
        quakeSequence = np.array(quakeWindow[["time.seconds", "magnitude", "longitude", "latitude", "depth"]])
        quakeSequence[:,0] = quakeSequence[:,0] - (i-W+1) * 24 * 60 * 60
    else:
        quakeSequence = np.array([])

    if len(predWindow) > 0:
        allX_maxMag.append(predWindow["magnitude"].max())
        allX_meanMag.append(predWindow["magnitude"].mean())
    else:
        allX_maxMag.append(0)
        allX_meanMag.append(0)

    allX_quakes.append(quakeSequence)
    allX_logN.append(np.log(len(predWindow) + 1)) #variable to predict

    allX_quakes_N.append(len(quakeSequence))
    
    allX_quakes_logN.append(np.log(len(quakeSequence) + 1))


allX_sunspots = np.array(allX_sunspots)

N = allX_sunspots.shape[0]
ssDistanceMatrix = np.zeros([N, N])

for i in range(N):
    squares = (allX_sunspots[i] - allX_sunspots)**2
    rowSums = np.sum(squares, axis=1)
    
    ssDistanceMatrix[i,:] = np.sqrt(rowSums)


slic = data[data["year"] < 2011]

timeStd = np.std(np.diff(slic["time.seconds"])) * args.tlambda
magnitudeStd = slic["magnitude"].std()
depthStd     = slic["depth"].std()
latitudeStd  = slic["latitude"].std()
longitudeStd = slic["longitude"].std()

baselineStds = [timeStd, magnitudeStd, longitudeStd, latitudeStd, depthStd]
baselineStds2 = [timeStd, magnitudeStd, (longitudeStd + latitudeStd) / 2, depthStd]


# Requires data1 and data2 given as matrices whose rows are
#   in the form (timestamp, magnitude, m2, m3, ...)
def editDistance(data1, data2, lambdas, lambdaDeletion=1):
    if len(data1) == 0:
        return len(data2)
    elif len(data2) == 0:
        return len(data1)
    
    allLen = len(data1) + len(data2)

    #print("{}-{}-{}".format(len(data1), len(data2), allLen), end="\t")
    
    lambdas = np.asarray(lambdas)
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)

    M = np.zeros([allLen, allLen])
    M[0:len(data1),0:len(data1)] = lambdaDeletion
    M[len(data1):allLen,len(data1):allLen] = lambdaDeletion

    #print(M.shape)
    
#    M[0:len(data1),0:len(data1)] = np.inf
#    M[len(data1):allLen,len(data1):allLen] = np.inf
#    M[np.diag_indices(allLen)] = [ lambdaDeletion for i in range(allLen) ]

    # Define shift costs
    for i in range(len(data1)):
        point1 = data1[i,:]
        
        diffs = np.abs(data2 - point1)
        
        # after this, diffs contain the c(i,j) for each j
        diffs = diffs @ lambdas
        
        M[i, len(data1):allLen] = diffs    
    
    row_ind, col_ind = linear_sum_assignment(M)
    
    #print(M)
    
    return M[row_ind,col_ind].sum()


baselineLambdas = [ 1 / i for i in baselineStds ]
baselineLambdas2 = [ 1 / i for i in baselineStds2 ]

def calculateDistances(idx):
    if idx % 50 == 0:
        print(idx, end=" ")

    N = len(allX_quakes)
    distances = []

    myX = allX_quakes[idx]
    
    for i in range(idx+1, N):
        #print(i, sep=" ")
        theirX = allX_quakes[i]
        distances.append(editDistance(myX, theirX, baselineLambdas))
        #distances.append(editDistance2(myX, theirX, baselineLambdas2))
    
    return distances


# allDistances = [ calculateDistances(i) for i in range(len(allX_quakes)) ]

try:
    mp.set_start_method('fork')
except Exception: pass

# Deleting some variables here to minimize memory usage by child processes
del data, slic, ss, ssDistanceMatrix

beg = time.time()

print("Beginning multiprocessed calculation.")
with mp.Pool(args.nthreads) as p:
    allDistances = p.map(calculateDistances, list(range(len(allX_quakes))), chunksize=1)

end = time.time()
print("Elapsed: ", end - beg)

N = len(allX_quakes)

distanceMatrix = np.zeros([N, N])

for i in range(N):
    distanceMatrix[i, (i+1):] = allDistances[i]
    distanceMatrix[(i+1):, i] = allDistances[i]

try:
    # This might fail if the user choses an output directory that is in an external HD
    np.save(os.path.join(args.outdir, f"{EXPERIMENT_NAME}.npy"), distanceMatrix)
except:
    # In that case, we save it in the current directory
    np.save(os.path.join("./", f"{EXPERIMENT_NAME}.npy"), distanceMatrix)