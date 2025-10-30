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
from data_loaders import load_cluster_dataset, EQTimeWindows, VALID_CLUSTER_REGIONS, pkldump

STRIDE=250 # Stride of the multiprocessed calculation
RANDOMIZE_ARGS = True # Whether to calculate edit distances in a randomized way

parser = argparse.ArgumentParser(prog='Calculator of Edit Distances For Clusterized Datasets')
parser.add_argument("--region",
        help="Region to analyze.",
        default="ja",
        choices=VALID_CLUSTER_REGIONS)
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
        default=30)
parser.add_argument("--outputw",
        help="Size of the time window for which a prediction is sought.",
        type=int,
        default=7)
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
parser.add_argument("--partial",
        help="Calculates edit distances to only some of the bases in the training set. " + \
              "Specify the number of bases with --partial-n",
        action='store_true')
parser.add_argument("--partial-n",
        help="When using --partial, the number of bases in the training set to consider.",
        type=int,
        default=500)
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
    f"distance-matrix-clustered",
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
EXPERIMENT_NAME = "-".join(EXPERIMENT_NAME)
print(f"Experiment name: {EXPERIMENT_NAME}")

# Requires data1 and data2 given as matrices whose rows are
#   in the form (timestamp, magnitude, m2, m3, ...)
def editDistance(data1, data2, lambdas, lambdaDeletion=1):
    if len(data1) == 0:
        return len(data2)
    elif len(data2) == 0:
        return len(data1)
    
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
    for i in range(len(data1)):
        point1 = data1[i,:]

        diffs = np.abs(data2 - point1)

        # after this, diffs contain the c(i,j) for each j
        diffs = diffs @ lambdas

        M[i, len(data1):allLen] = diffs    

    row_ind, col_ind = linear_sum_assignment(M)

    return M[row_ind,col_ind].sum()

def calculateDistances(idx, allX_quakes, valid_idxs):
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


data = load_cluster_dataset(args.region, args.minmag)
allDistMatrices = []

for df in data:
    eqtw = EQTimeWindows(df, args.inputw, args.outputw, args.nthreads)

    baselineStds = eqtw.getBaselineStds(args.tlambda)
    baselineLambdas = [ 1 / i for i in baselineStds ]

    quakes = eqtw.x_quakes[0]

    if args.partial:
        valid_idxs = random.sample(range(len(quakes) // 2), k=args.partial_n)
        #print(sorted(valid_idxs))
        valid_idxs = set(valid_idxs)
    else:
        valid_idxs = set(range(len(quakes)))

    def calculateDistances_d(idx):
        return calculateDistances(idx, quakes, valid_idxs)

    # allDistances = [ calculateDistances(i) for i in range(len(allX_quakes)) ]

    try:
        mp.set_start_method('fork')
    except Exception: pass

    allArgs = list(range(0,len(quakes)**2,STRIDE))

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
        allDistances = list(tqdm.tqdm(p.imap_unordered(calculateDistances_d, allArgs, chunksize=1), total=len(allArgs), smoothing=0.1))
    end = time.time()
    print("Elapsed: ", end - beg)

    N = len(quakes)

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

    allDistMatrices.append(distanceMatrix)

# Once all matrices are obtained, we save it.
try:
    # This might fail if the user choses an output directory that is in an external HD
    fname = os.path.join(args.outdir, f"{EXPERIMENT_NAME}.pickle")
    if os.path.exists(fname):
        print("File already exists! Will preprend a random identifier to it.")
        fname = os.path.join(args.outdir, f"{EXPERIMENT_NAME}-{os.urandom(4).hex()}.pickle")
    pkldump(allDistMatrices, fname)
except:
    # In that case, we save it in the current directory
    print("Could not save in the specified folder. Saving to current folder instead.")
    fname = os.path.join("./", f"{EXPERIMENT_NAME}.pickle")
    if os.path.exists(fname):
        print("File already exists! Will preprend a random identifier to it.")
        fname = os.path.join("./", f"{EXPERIMENT_NAME}-{os.urandom(4).hex()}.pickle")
    pkldump(allDistMatrices, fname)