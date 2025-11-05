import numpy as np
import sys
import tqdm
from forecasting_rbf import predict_rbf
from data_loaders import * # For testing

def experiments_rbf(distMat, all_predictor, all_expected, trainSize, eps, numIter=100, numBases=100):
    experiment = {
        "correlation": [],
        "predicted": []
    }

    for i in tqdm.tqdm(range(numIter)):
        print(i, end=" ", file=sys.stderr)

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

        predicted = predict_rbf(all_predictor, selectDistances, trainSize, eps)

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

if __name__ == "__main__":
    data = load_dataset("ja")
    distMat = np.load("/media/mathjs/HD-ADU3/distance-matrix-ja-minmag0-inputw7-outputw1-tlambda1.npy")

    eqtw = EQTimeWindows(data, 7, 1, nthreads = 22)

    trainSize = distMat.shape[0]//2
    trainMat = distMat[:trainSize,:trainSize]
    eps = 2 * np.mean(trainMat[trainMat > 0])**2

    logN = [ len(i) for i in eqtw.y_quakes[0][1:] ]

    experiment = experiments_rbf(
        distMat,
        logN,
        logN,
        trainSize,
        eps,
        numIter=20)
    
    print(experiment)
    print(np.mean(experiment["correlation"]))