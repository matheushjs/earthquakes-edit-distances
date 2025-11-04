import numpy as np
import sys
import tqdm
from forecasting_rbf import predict_rbf

def experiments_rbf(distMat, all_expected, trainSize, eps, numIter=100, numBases=100):
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

        predicted = predict_rbf(selectDistances, trainSize, eps)

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
