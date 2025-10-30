import numpy as np
import pandas as pd
import datetime as dt
import pickle
import time
import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

VALID_REGIONS = ["ja", "gr", "nz", "jma", "toho", "well", "stil", "jmatoho"]
VALID_CLUSTER_REGIONS = ["ja", "nz", "gr"]
DF_COLUMNS_MAIN = ["time.seconds", "magnitude", "longitude", "latitude", "depth"]
TS_INDEX = 0
MAG_INDEX = 1
LON_INDEX = 2
LAT_INDEX = 3
DEP_INDEX = 4

def pkldump(obj, file):
    with open(file, "wb") as fp:
        pickle.dump(obj, fp)

def pklload(file):
    with open(file, "rb") as fp:
        return pickle.load(fp)

def add_time_columns(data):
    initDate = dt.datetime(2000, 1, 1)
    newCol = []
    newCol2 = []

    data = data.copy() # Just to avoid changing the callee data

    for i in range(len(data)):
        datum = data.iloc[i,:]
        dateObj = dt.datetime(int(datum["year"]), int(datum["month"]), int(datum["day"]))
        ndays = (dateObj - initDate).days

        try:
            dateObj = dt.datetime(int(datum["year"]), int(datum["month"]), int(datum["day"]), int(datum["hour"]), int(datum["minute"]), int(datum["second"]))
        except Exception:
            dateObj = dt.datetime(int(datum["year"]), int(datum["month"]), int(datum["day"]), int(datum["hour"]), int(datum["minute"]))

        nsecs = dateObj.timestamp() - initDate.timestamp()
        
        newCol.append(ndays)
        newCol2.append(nsecs)

    data["day.number"] = newCol
    data["time.seconds"] = newCol2

    return data

def load_dataset(region, minmag=0):
    if region not in VALID_REGIONS:
        raise Exception(f"Invalid region: {region}.")

    regionToDatafile = {
        "ja": "../japan.csv",
        "toho": "../japan.csv",
        "gr": "../greece.csv",
        "stil": "../greece.csv",
        "nz": "../newzealand.csv",
        "well": "../newzealand.csv",
        "jma": "../jma-japan-earthquakes-mag2-dayNumbers.csv",
        "jmatoho": "../jma-japan-earthquakes-mag2-dayNumbers.csv"
    }

    data = pd.read_csv(regionToDatafile[region])

    if minmag == 0 and (region == "jma" or region == "jmatoho"):
        minmag = 2.7

    data = data[data["magnitude"] >= minmag]
    data = data.query("year <= 2021").query("year < 2021 or month < 9")
    data = data.copy().reset_index(drop=True) # Ensures we are not working with a pandas slice

    #data = add_time_columns(data)

    if region == "toho" or region == "jmatoho":
        points = data[["longitude", "latitude"]].copy()
        centroid = np.array([143, 37.9])
        idx = np.sqrt(np.sum((points - centroid)**2, axis=1)) < 2.8
        data = data[idx].copy().reset_index(drop=True)
    elif region == "well":
        points = data[["longitude", "latitude"]].copy()
        centroid = np.array([174.776230, -41.286461])
        idx = np.sqrt(np.sum((points - centroid)**2, axis=1)) < 2.8
        data = data[idx].copy().reset_index(drop=True)
    elif region == "stil":
        points = data[["longitude", "latitude"]].copy()
        centroid = np.array([21.25, 37.5])
        idx = np.sqrt(np.sum((points - centroid)**2, axis=1)) < 2
        data = data[idx].copy().reset_index(drop=True)
    
    return data

def load_cluster_dataset(region, minmag=0, plot=False):
    if region not in VALID_CLUSTER_REGIONS:
        raise Exception(f"Invalid region: {region}.")

    regionToDatafile = {
        "ja": "../japan-clusterized-dataframes.pickle",
        "nz": "../newzealand-clusterized-dataframes.pickle",
        "gr": "../greece-clusterized-dataframes.pickle"
    }

    data = pklload(regionToDatafile[region])

    for i in range(len(data)):
        d = data[i]
        d = d[d["magnitude"] >= minmag]
        d = d.query("`day.number` < 7914") #d.query("year <= 2021").query("year < 2021 or month < 9")
        d = d.copy().reset_index(drop=True) # Ensures we are not working with a pandas slice
        data[i] = d

    # Sort based on minimum longitude
    data = sorted(data, key=lambda x: x.longitude.min())

    if plot:
        cmap = sns.color_palette("Set2")

        for i, slic in enumerate(data):
            mlat = slic["latitude"].median()
            mlon = slic["longitude"].median()
            
            plt.scatter(slic["longitude"], slic["latitude"], color=cmap[i % len(cmap)])
            
            plt.text(mlon, mlat, str(i))

        plt.show()

    return data

# Collects the set of earthquakes in each time window of size windowSize
# For now we assume a stride of 1 day
# Days that could not be calculated are returned as None
# We do not calculate time windows that are not fully contained in the dataframe's range
# 'lastDayNumber' is the last day number up to which to process the dataframe.
make_sets_of_eqs_mp_get_eqs_data = {
    "data": None,
    "windowSize": None,
    "lastDayNumber": None,
    "firstDayNumber": None
}
def make_sets_of_eqs_mp_get_eqs(i):
    data = make_sets_of_eqs_mp_get_eqs_data["data"]
    windowSize = make_sets_of_eqs_mp_get_eqs_data["windowSize"]
    lastDayNumber = make_sets_of_eqs_mp_get_eqs_data["lastDayNumber"]
    firstDayNumber = make_sets_of_eqs_mp_get_eqs_data["firstDayNumber"]
    dayNumbers = data["day.number"].to_numpy()

    windowFirstDN = i - windowSize + 1
    windowLastDN  = i

    if windowFirstDN < firstDayNumber or windowLastDN > lastDayNumber:
        return None

    quakeWindow = data[ (dayNumbers >= windowFirstDN) * (dayNumbers <= windowLastDN) ]
    if len(quakeWindow) > 0:
        quakeSequence = np.array(quakeWindow[DF_COLUMNS_MAIN])
        quakeSequence[:,0] = quakeSequence[:,0] - (i-windowSize+1) * 24 * 60 * 60
    else:
        quakeSequence = np.array([])

    return quakeSequence
def make_sets_of_eqs(data, windowSize, nThreads, lastDayNumber=7913, firstDayNumber=0):
    make_sets_of_eqs_mp_get_eqs_data["data"]            = data
    make_sets_of_eqs_mp_get_eqs_data["windowSize"]      = windowSize
    make_sets_of_eqs_mp_get_eqs_data["lastDayNumber"]   = lastDayNumber
    make_sets_of_eqs_mp_get_eqs_data["firstDayNumber"]  = firstDayNumber

    allArgs = range(firstDayNumber, lastDayNumber+1)
    with mp.Pool(nThreads) as p:
        allQuakeSequences = list(tqdm.tqdm(p.imap(make_sets_of_eqs_mp_get_eqs, allArgs, chunksize=250), total=len(allArgs), smoothing=0.1))

    return allQuakeSequences

def quake_sequence_basic_stats_mp_get_stats(quakes):
    maxMag = quakes[:,MAG_INDEX].max() if len(quakes) > 0 else 0
    meanMag = quakes[:,MAG_INDEX].mean() if len(quakes) > 0 else 0
    N = len(quakes)
    logN = np.log(len(quakes) + 1)

    return [ maxMag, meanMag, N, logN ]
def quake_sequence_basic_stats(quakeSequence, nThreads):
    allArgs = quakeSequence
    with mp.Pool(nThreads) as p:
        allStats = list(tqdm.tqdm(p.imap(quake_sequence_basic_stats_mp_get_stats, allArgs, chunksize=250), total=len(allArgs), smoothing=0.1))

    return pd.DataFrame(allStats, columns=["maxMag", "meanMag", "N", "logN"])

class EQTimeWindows:
    def __init__(self, data, inputw=7, outputw=1, nthreads=1):
        try:
            for col in ["day.number", "magnitude", "latitude", "longitude", "depth", "time.seconds"]:
                data[col]
        except:
            raise Exception("Dataframe does not have the required columns.")

        if not isinstance(inputw, list):
            inputw = [inputw]
        if not isinstance(outputw, list):
            outputw = [outputw]

        self.inputw = inputw
        self.outputw = outputw
        self.data = data
        self.nthreads = nthreads

        # prefix 'x' means the variable refers to the X space, the independent variable space (the previous time windows)
        # prefix 'y' means the dependent variable Y space, often the next-window features
        self.x_quakes  = [ make_sets_of_eqs(self.data, windowSize, self.nthreads) for windowSize in inputw ]
        self.y_quakes  = [ make_sets_of_eqs(self.data, windowSize, self.nthreads) for windowSize in outputw ]

        self.x_quakes, self.y_quakes = self._trimQuakes(self.x_quakes, self.y_quakes)

        self.x_stats = [ quake_sequence_basic_stats(seq, self.nthreads) for seq in self.x_quakes ]
        self.y_stats = [ quake_sequence_basic_stats(seq, self.nthreads) for seq in self.y_quakes ]

    def getBaselineStds(self, tlambda):
        df = self.data
        slic = df[df["day.number"] < 4018] # "year" < 2011

        timeStd      = np.std(np.diff(slic["time.seconds"])) * tlambda
        magnitudeStd = df["magnitude"].std()
        depthStd     = df["depth"].std()
        latitudeStd  = df["latitude"].std()
        longitudeStd = df["longitude"].std()

        return [timeStd, magnitudeStd, longitudeStd, latitudeStd, depthStd]

    # Trims the quake sequences to remove all the trailing and leading None objects
    def _trimQuakes(self, xquakes, yquakes):
        allQuakes = xquakes + yquakes

        # Gets the indices for which an earthquake time window exists in each quakeSeries
        # in xquakes AND yquakes simultaneously
        indices = np.argwhere(np.all([ [ i is not None for i in quakes ] for quakes in allQuakes ], axis=0)).ravel()

        xquakes = [ [ quakes[idx] for idx in indices ] for quakes in xquakes ]
        yquakes = [ [ quakes[idx] for idx in indices ] for quakes in yquakes ]

        return xquakes, yquakes