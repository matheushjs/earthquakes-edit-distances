import numpy as np
import sys
import tqdm
from forecasting_rbf import predict_rbf
from data_loaders import * # For testing

experiments_rbf_mp_data = {
    "distMat": None,
    "all_predictor": None,
    "all_expected": None,
    "trainSize": None,
    "eps": None,
    "numBases": None
}
def experiments_rbf_mp(i):
    distMat = experiments_rbf_mp_data["distMat"]
    all_predictor = experiments_rbf_mp_data["all_predictor"]
    all_expected = experiments_rbf_mp_data["all_expected"]
    trainSize = experiments_rbf_mp_data["trainSize"]
    eps = experiments_rbf_mp_data["eps"]
    numBases = experiments_rbf_mp_data["numBases"]

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

    if i == 0:
        retval = [np.corrcoef(predicted, mmags)[0,1], predicted]
    else:
        retval = [np.corrcoef(predicted, mmags)[0,1], None]

    return retval

def experiments_rbf(distMat, all_predictor, all_expected, trainSize, eps, numIter=100, numBases=100, nThreads=1):
    experiments_rbf_mp_data["distMat"] = distMat
    experiments_rbf_mp_data["all_predictor"] = all_predictor
    experiments_rbf_mp_data["all_expected"] = all_expected
    experiments_rbf_mp_data["trainSize"] = trainSize
    experiments_rbf_mp_data["eps"] = eps
    experiments_rbf_mp_data["numBases"] = numBases

    allArgs = range(numIter)
    with mp.Pool(nThreads) as p:
        results = list(tqdm.tqdm(p.imap_unordered(experiments_rbf_mp, allArgs, chunksize=1),
                                 total=len(allArgs),
                                 smoothing=0.1,
                                 desc="RBF predictions"))

    experiment = {}
    experiment["correlation"] = np.array([ i[0] for i in results ])
    experiment["predicted"] = list(filter(lambda x: x is not None, [ i[1] for i in results ]))

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
        numIter=100,
        nThreads=4)
    
    print(experiment)
    print(np.mean(experiment["correlation"]))