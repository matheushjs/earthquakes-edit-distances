import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
import pickle
import datetime as dt
import copy
import tqdm
import seaborn as sns
import multiprocessing as mp
import time
import argparse, sys, os, random
from data_loaders import load_dataset, VALID_REGIONS

STRIDE=250 # Stride of the multiprocessed calculation
RANDOMIZE_ARGS = True # Whether to calculate edit distances in a randomized way

parser = argparse.ArgumentParser(prog='Calculator of Edit Distances')
parser.add_argument("--region",
        help="Region to analyze.",
        default="ja",
        choices=VALID_REGIONS)
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
parser.add_argument("--dry-run",
        help="Calculate only a small portion of the edit distances, just for testing." ,
        action='store_true')
parser.add_argument("--dummy-dists",
        help="Do not calculate edit distances. Used for testing correctness of the code." ,
        action='store_true')
parser.add_argument("--calc-time",
        help="Return a matrix of execution times, rather than edit distances." ,
        action='store_true')
parser.add_argument("--partial",
        help="Calculates edit distances to only some of the bases in the training set. " + \
              "Specify the number of bases with --partial-n",
        action='store_true')
parser.add_argument("--partial-n",
        help="When using --partial, the number of bases in the training set to consider.",
        type=int,
        default=500)
parser.add_argument("--limit-windows",
        help="Limits the number of earthquakes in each window to the X-percentile of the window sizes distribution.",
        action='store_true')
parser.add_argument("--limit-windows-p",
        help="Percentile to which window sizes should be truncated to.",
        type=float,
        default=0.8)
args = parser.parse_args()

if args.dry_run and args.dummy_dists or args.partial and args.dummy_dists:
    print("Cannot use --dummy-dists with --dry-run or --partial.")
    sys.exit(1)

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
if args.dry_run:
    EXPERIMENT_NAME += ["dryrun"]
if args.partial:
    EXPERIMENT_NAME += [f"partial{args.partial_n}"]
if args.dummy_dists:
    EXPERIMENT_NAME += ["dummydists"]
if args.calc_time:
    EXPERIMENT_NAME += ["times"]
EXPERIMENT_NAME = "-".join(EXPERIMENT_NAME)
print(f"Experiment name: {EXPERIMENT_NAME}")

def pkldump(obj, file):
    with open(file, "wb") as fp:
        pickle.dump(obj, fp)

def pklload(file):
    with open(file, "rb") as fp:
        return pickle.load(fp)

# Requires data1 and data2 given as matrices whose rows are
#   in the form (timestamp, magnitude, m2, m3, ...)
def editDistance(data1, data2, lambdas, lambdaDeletion=1):
    if len(data1) == 0:
        return len(data2) if not args.calc_time else 0.0
    elif len(data2) == 0:
        return len(data1) if not args.calc_time else 0.0
    
    allLen = len(data1) + len(data2)
    
    lambdas = np.asarray(lambdas)
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)

    M = np.zeros([allLen, allLen])
    M[0:len(data1),0:len(data1)] = lambdaDeletion
    M[len(data1):allLen,len(data1):allLen] = lambdaDeletion
    
#    M[0:len(data1),0:len(data1)] = np.inf
#    M[len(data1):allLen,len(data1):allLen] = np.inf
#    M[np.diag_indices(allLen)] = [ lambdaDeletion for i in range(allLen) ]

    # Define shift costs
    timer = time.time()
    for i in range(len(data1)):
        point1 = data1[i,:]

        diffs = np.abs(data2 - point1)

        # after this, diffs contain the c(i,j) for each j
        diffs = diffs @ lambdas

        M[i, len(data1):allLen] = diffs    

    row_ind, col_ind = linear_sum_assignment(M)
    elapsed = time.time() - timer

    if args.calc_time:
        return elapsed

    return M[row_ind,col_ind].sum()

data = load_dataset(args.region, args.minmag)


allX_quakes = []
allX_maxMag = []
allX_meanMag = []
allX_logN = []

allX_quakes_N = []
allX_quakes_logN = []
#allX_seismicity = []

allY_dayNumbers = []

# Window size
W = args.inputw

# Prediction window
PRED_WINDOW = args.outputw

dayNumbers = data["day.number"] # To make things faster, save it in a variable
maxDayNumber = (dt.datetime(2021, 8, 31) - dt.datetime(2000, 1, 1)).days
# NOTE: day.number of 2000/01/01 is 0, so we do not need to add 1 to maxDayNumber

for i in range(W-1, maxDayNumber + 1 - PRED_WINDOW):
    quakeWindow = data[ (dayNumbers > i - W) * (dayNumbers <= i) ]
    predWindow  = data[ (dayNumbers > i) * (dayNumbers <= i+PRED_WINDOW) ]
    allY_dayNumbers.append(i + 1)

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


slic = data[data["year"] < 2011]

timeStd      = np.std(np.diff(slic["time.seconds"])) * args.tlambda
magnitudeStd = slic["magnitude"].std()
depthStd     = slic["depth"].std()
latitudeStd  = slic["latitude"].std()
longitudeStd = slic["longitude"].std()

baselineStds = [timeStd, magnitudeStd, longitudeStd, latitudeStd, depthStd]
#baselineStds2 = [timeStd, magnitudeStd, (longitudeStd + latitudeStd) / 2, depthStd]

baselineLambdas = [ 1 / i for i in baselineStds ]
#baselineLambdas2 = [ 1 / i for i in baselineStds2 ]

def calculateDistances(idx):
    if idx % 50 == 0:
        print(idx, end=" ")

    N = len(allX_quakes)
    distances = []

    myX = allX_quakes[idx]
    
    for i in range(idx+1, N):
        theirX = allX_quakes[i]
        if args.dry_run and random.random() > 0.00005:
            distances.append(-1)
        else:
            distances.append(editDistance(myX, theirX, baselineLambdas) if not args.dummy_dists else float(idx*10000 + i) )
        #distances.append(editDistance2(myX, theirX, baselineLambdas2))
    
    return distances

if args.partial:
    valid_idxs = random.sample(range(len(allX_quakes) // 2), k=args.partial_n)
    #print(sorted(valid_idxs))
    valid_idxs = set(valid_idxs)
else:
    valid_idxs = set(range(len(allX_quakes)))

def calculateDistances2(idx):
    N = len(allX_quakes)

    #if idx % (STRIDE*500) == 0:
    #    print(f"{idx / STRIDE / (N*N / STRIDE):.2f}", end=" ")

    distances = []

    for i in range(idx, idx+STRIDE):
        idx_i = i // N
        idx_j = i % N

        if i >= N*N:
            break

        if idx_i >= idx_j:
            # We make it zero so we can complete the matrix later
            # By simply adding the transpose to itself
            distances.append(0)
            continue
    
        myX    = allX_quakes[idx_i]
        theirX = allX_quakes[idx_j]
        if args.dry_run and random.random() > 0.00005:
            distances.append(-1)
        elif idx_i in valid_idxs or idx_j in valid_idxs:
            distances.append(editDistance(myX, theirX, baselineLambdas) if not args.dummy_dists else float(idx_i*10000 + idx_j) )
            #distances.append(idx_j)
        else:
            distances.append(-1)
        #distances.append(editDistance2(myX, theirX, baselineLambdas2))
    
    return idx, distances

# allDistances = [ calculateDistances(i) for i in range(len(allX_quakes)) ]

try:
    mp.set_start_method('fork')
except Exception: pass

# Deleting some variables here to minimize memory usage by child processes
del data, slic

allArgs = list(range(0,len(allX_quakes)**2,STRIDE))

if RANDOMIZE_ARGS:
    allArgs_idxRand = list(range(len(allArgs))) 
    np.random.shuffle(allArgs_idxRand) # 4 2 0 3 1
    allArgs_idxRand_inv = np.zeros(len(allArgs), dtype=int)
    for i in range(len(allArgs)):
        allArgs_idxRand_inv[allArgs_idxRand[i]] = i  # 2 4 1 3 0
    
    allArgs = [ allArgs[i] for i in allArgs_idxRand ]

beg = time.time()
print("Beginning multiprocessed calculation.")
with mp.Pool(args.nthreads) as p:
    #allDistances = p.map(calculateDistances2, allArgs, chunksize=1)
    allDistances = list(tqdm.tqdm(p.imap_unordered(calculateDistances2, allArgs, chunksize=1), total=len(allArgs), smoothing=0.1))
end = time.time()
print("Elapsed: ", end - beg)

# if RANDOMIZE_ARGS:
#     allArgs      = [ allArgs[i] for i in allArgs_idxRand_inv ]
#     allDistances = [ allDistances[i] for i in allArgs_idxRand_inv ]

N = len(allX_quakes)

distanceMatrix = np.zeros(N**2)

# for i, distances in zip(allArgs, allDistances):
for i, distances in allDistances:
    if len(distances) == STRIDE:
        distanceMatrix[i:(i+STRIDE)] = distances
    else:
        distanceMatrix[i:] = distances

distanceMatrix = distanceMatrix.reshape(N, N)
distanceMatrix = distanceMatrix + distanceMatrix.T # Complete the lower triangle

if args.dummy_dists:
    correctMat = np.array([ [ j*10000 + i for i in range(N) ] for j in range(N) ])
    correctMat[np.tril_indices(N)] = 0
    correctMat[np.diag_indices(N)] = 0
    correctMat = correctMat + correctMat.T
    
    if np.all(correctMat == distanceMatrix):
        print("Distance matrix was filled correctly.")
    else:
        print("ERROR: Distance matrix was NOT filled correctly.")

# if args.partial:
#     distanceMatrix = distanceMatrix[:,np.array([i for i in valid_idxs])]

try:
    # This might fail if the user choses an output directory that is in an external HD
    fname = os.path.join(args.outdir, f"{EXPERIMENT_NAME}.npy")
    if os.path.exists(fname):
        print("File already exists! Will preprend a random identifier to it.")
        fname = os.path.join(args.outdir, f"{EXPERIMENT_NAME}-{os.urandom(4).hex()}.npy")
    np.save(fname, distanceMatrix)
except:
    # In that case, we save it in the current directory
    print("Could not save in the specified folder. Saving to current folder instead.")
    fname = os.path.join("./", f"{EXPERIMENT_NAME}.npy")
    if os.path.exists(fname):
        print("File already exists! Will preprend a random identifier to it.")
        fname = os.path.join("./", f"{EXPERIMENT_NAME}-{os.urandom(4).hex()}.npy")
    np.save(fname, distanceMatrix)