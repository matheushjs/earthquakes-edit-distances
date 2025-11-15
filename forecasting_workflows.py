import numpy as np
import sklearn.metrics as metrics
import torch
import multiprocessing as mp
import os
import tqdm
from forecasting_rbf import predict_rbf
from forecasting_ff_nn import predict_ff_nn
from data_loaders import * # For testing

def optimize_thresholds(predicted, mmags, grid_steps=50):
    """
    Performs a 2D grid search to find the best alpha and beta thresholds.
    
    Args:
        predicted (np.array): Predicted log-number of earthquakes.
        mmags (np.array): Actual max magnitudes (ground truth).
        grid_steps (int): The resolution of the grid for both alpha and beta.

    Returns:
        A dictionary containing the best parameters and the full grid data.
    """
    
    epsilon = 1e-6
    alpha_grid = np.linspace(np.min(predicted) - epsilon, 
                             np.max(predicted) + epsilon, 
                             grid_steps)
    
    beta_min = 4.5
    beta_max = np.max(mmags)
    
    if beta_max < beta_min:
        print(f"Warning: Max magnitude in data ({beta_max}) is less than beta_min (4.5).")
        return None

    beta_grid = np.linspace(beta_min, beta_max + epsilon, grid_steps)

    # scores_grid will store the MCC for each (alpha, beta) pair
    # Rows will correspond to alpha, columns to beta
    scores_grid = np.zeros((grid_steps, grid_steps))
    
    best_mcc = -1  # MCC ranges from -1 to 1
    best_alpha = None
    best_beta = None

    for i, alpha in enumerate(alpha_grid):
        for j, beta in enumerate(beta_grid):
            
            # Apply the classification rule
            # y_pred: We predict "True" (event) if predicted log-num > alpha
            y_pred = (predicted > alpha)
            
            # y_true: The "True" event is if the actual magnitude > beta
            y_true = (mmags > beta)
            
            # Calculate score
            tp = np.sum((y_true) * (y_pred))
            fp = np.sum((~y_true) * (y_pred))
            fn = np.sum((y_true) * (~y_pred))
            tn = np.sum((~y_true) * (~y_pred))
            if tp > 0 and fp > 0 and tn > 0 and fn > 0:
                odds = np.log(tp / fp) + np.log(tn / fn)
                mcc = metrics.matthews_corrcoef(y_true, y_pred)
            else:
                odds = None
                mcc = 0

            scores_grid[i, j] = mcc
            
            if mcc > best_mcc:
                best_mcc = mcc
                best_alpha = alpha
                best_beta = beta
                best_odds = odds
                
    return {
        "best_alpha": best_alpha,
        "best_beta": best_beta,
        "best_mcc": best_mcc,
        "best_odds": best_odds
        # "alpha_grid": alpha_grid,
        # "beta_grid": beta_grid,
        # "scores_grid": scores_grid
    }

experiments_rbf_mp_data = {
    "distMat": None,
    "all_predictor": None,
    "all_expected": None,
    "trainSize": None,
    "eps": None,
    "numBases": None,
    "pseudoinv": None
}
def experiments_rbf_mp(i):
    np.random.seed()

    distMat = experiments_rbf_mp_data["distMat"]
    all_predictor = experiments_rbf_mp_data["all_predictor"]
    all_expected = experiments_rbf_mp_data["all_expected"]
    trainSize = experiments_rbf_mp_data["trainSize"]
    eps = experiments_rbf_mp_data["eps"]
    numBases = experiments_rbf_mp_data["numBases"]
    pseudoinv = experiments_rbf_mp_data["pseudoinv"]

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

    predicted, real = predict_rbf(all_predictor, selectDistances, trainSize, eps, pseudoinv=pseudoinv)

    mmags = np.array(all_expected)[trainSize:]

    # # Thresholds
    # a = 6
    # b = 3.2

    # tp = np.sum((mmags >  a) * (predicted >  b))
    # fp = np.sum((mmags <= a) * (predicted >  b))
    # fn = np.sum((mmags >  a) * (predicted <= b))
    # tn = np.sum((mmags <= a) * (predicted <= b))

    metrics = optimize_thresholds(predicted, mmags)

    if i == 0:
        retval = [np.corrcoef(predicted, mmags)[0,1], predicted, metrics]
    else:
        retval = [np.corrcoef(predicted, mmags)[0,1], None, metrics]

    return retval

def experiments_rbf(distMat, all_predictor, all_expected, trainSize, eps, numIter=100, numBases=100, pseudoinv=False, nThreads=1):
    experiments_rbf_mp_data["distMat"] = distMat
    experiments_rbf_mp_data["all_predictor"] = all_predictor
    experiments_rbf_mp_data["all_expected"] = all_expected
    experiments_rbf_mp_data["trainSize"] = trainSize
    experiments_rbf_mp_data["eps"] = eps
    experiments_rbf_mp_data["numBases"] = numBases
    experiments_rbf_mp_data["pseudoinv"] = pseudoinv

    # Prevent each thread from creating too many other threads
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["NUMEXPR_NUM_THREADS"] = "2"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "2"

    allArgs = range(numIter)
    with mp.Pool(nThreads) as p:
        results = list(tqdm.tqdm(p.imap_unordered(experiments_rbf_mp, allArgs, chunksize=1),
                                 total=len(allArgs),
                                 smoothing=0.1,
                                 desc="RBF predictions"))

    del os.environ["OMP_NUM_THREADS"]
    del os.environ["OPENBLAS_NUM_THREADS"]
    del os.environ["MKL_NUM_THREADS"]
    del os.environ["NUMEXPR_NUM_THREADS"]
    del os.environ["VECLIB_MAXIMUM_THREADS"]

    experiment = {}
    experiment["correlation"] = np.array([ i[0] for i in results ])
    experiment["predicted"] = list(filter(lambda x: x is not None, [ i[1] for i in results ]))
    experiment["metrics"] = np.array([ i[2] for i in results ])

    return experiment

experiments_ff_nn_mp_data = {
    "distMat": None,
    "seisFeatures": None,
    "all_predictor": None,
    "all_expected": None,
    "trainSize": None,
    "numBases": None,
    "si_activation": None
}
def experiments_ff_nn_mp(i):
    np.random.seed(i + int(time.time()))
    torch.manual_seed(i + int(time.time()))

    distMat = experiments_ff_nn_mp_data["distMat"]
    seisFeatures = experiments_ff_nn_mp_data["seisFeatures"]
    all_predictor = experiments_ff_nn_mp_data["all_predictor"]
    all_expected = experiments_ff_nn_mp_data["all_expected"]
    trainSize = experiments_ff_nn_mp_data["trainSize"]
    numBases = experiments_ff_nn_mp_data["numBases"]
    si_activation = experiments_ff_nn_mp_data["si_activation"] 

    predicted, testY, corr, mse = predict_ff_nn(all_predictor, trainSize, distMat=distMat, seisFeatures=seisFeatures,
                      earlyStoppingPatience=50, lr=0.01, verbose=False, numBases=numBases,
                      si_activation=si_activation, log_steps=1, eval_steps=10, batch_size=128)

    metrics = optimize_thresholds(predicted, all_expected[trainSize:])

    return predicted, testY, corr, mse, metrics

def experiments_ff_nn(all_predictor, all_expected, trainSize,
        distMat=None, seisFeatures=None, numBases=100, si_activation="relu",
        numIter=10, nThreads=1
):
    experiments_ff_nn_mp_data.update({
        "distMat": distMat,
        "seisFeatures": seisFeatures,
        "all_predictor": all_predictor,
        "all_expected": all_expected,
        "trainSize": trainSize,
        "numBases": numBases,
        "si_activation": si_activation
    })

    allArgs = range(numIter)
    with mp.Pool(nThreads) as p:
        results = list(tqdm.tqdm(p.imap_unordered(experiments_ff_nn_mp, allArgs, chunksize=1),
                                 total=len(allArgs),
                                 smoothing=0.1,
                                 desc="FF-NN predictions"))

    experiment = {
        "predicted": [],
        "real": [],
        "corr": [],
        "corr_expected": [],
        "mse": [],
        "metrics": []
    }

    for r in results:
        experiment["predicted"].append(r[0])
        experiment["real"].append(r[1])
        experiment["corr"].append(r[2])
        experiment["mse"].append(r[3])
        experiment["metrics"].append(r[4])
        experiment["corr_expected"].append(np.corrcoef(r[0], all_expected[trainSize:])[1,0])

    return experiment

if __name__ == "__main__":
    data = load_dataset("ja")
    distMat = np.load("/media/mathjs/HD-ADU3/distance-matrix-ja-minmag0-inputw7-outputw1-tlambda1.npy")

    eqtw = EQTimeWindows(data, 7, 1, nthreads = 22)

    trainSize = distMat.shape[0]//2
    trainMat = distMat[:trainSize,:trainSize]
    eps = 2 * np.mean(trainMat[trainMat > 0])**2

    logN = [ np.log(len(i) + 1) for i in eqtw.y_quakes[0][1:] ]
    mmags = eqtw.getXQuakesMaxMag()[0]

    # experiment = experiments_rbf(
    #     distMat,
    #     logN,
    #     logN,
    #     trainSize,
    #     eps,
    #     numIter=100,
    #     nThreads=20)

    experiment = experiments_ff_nn(logN, mmags[1:], trainSize, distMat=distMat, numIter=10, nThreads=5)

    print(experiment)
    print(np.mean(experiment["corr_expected"]))