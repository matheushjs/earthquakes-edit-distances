import numpy as np
import pandas as pd
import datetime as dt
import pickle
import time
import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
from os import path

VALID_REGIONS = ["ja", "gr", "nz", "jma", "toho", "well", "stil", "jmatoho"]
VALID_CLUSTER_REGIONS = ["ja", "nz", "gr"]
DF_COLUMNS_MAIN = ["time.seconds", "magnitude", "longitude", "latitude", "depth"]
TS_INDEX = 0
MAG_INDEX = 1
LON_INDEX = 2
LAT_INDEX = 3
DEP_INDEX = 4
MEMOIZATION_DIR = "/media/mathjs/HD-ADU3/caching/"

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
        allQuakeSequences = list(tqdm.tqdm(p.imap(make_sets_of_eqs_mp_get_eqs, allArgs, chunksize=250),
                                           total=len(allArgs),
                                           smoothing=0.1,
                                           desc="Making EQ sets"))

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
        allStats = list(tqdm.tqdm(p.imap(quake_sequence_basic_stats_mp_get_stats, allArgs, chunksize=250),
                                  total=len(allArgs),
                                  smoothing=0.1,
                                  desc="EQ basic stats"))

    return pd.DataFrame(allStats, columns=["maxMag", "meanMag", "N", "logN"])

def TValue(quakes, threshold):
    if len(quakes) == 0:
        return np.nan # We will later substitute by the maximum observed TValue

    mags = np.array([ i[MAG_INDEX] for i in quakes ])
    tss  = np.array([ i[TS_INDEX] for i in quakes ])
    idx = mags > threshold

    tss = tss[idx]

    if len(tss) < 2:
        return np.nan

    return tss[-1] - tss[0]

def meanMag(quakes):
    if len(quakes) == 0:
        return 0

    mags = np.array([ i[MAG_INDEX] for i in quakes ])

    return np.mean(mags)

def energyRate(quakes, windowLength):
    if len(quakes) == 0:
        return 0

    mags = np.array([ i[MAG_INDEX] for i in quakes ])

    return np.sum(np.sqrt(10**(mags*1.5 + 11.8))) / windowLength

def grLaw(quakes):
    if len(quakes) < 2:
        return [0, 0, np.nan, 0] #nan will be substituted by the largest value in the training set

    mags = np.array([ i[MAG_INDEX] for i in quakes ])
    N = np.array([ np.sum(mags >= mags[i]) for i in range(len(mags)) ])

    divisor = ( np.sum(mags)**2 - len(mags)*np.sum(mags**2) )
    if divisor == 0:
        return [0, 0, np.nan, 0]

    b = (len(mags) * np.sum(mags * N) - np.sum(mags) * np.sum(N)) / divisor
    a = np.sum(np.log10(N) + b*mags) / len(mags)
    eta = np.sum( (np.log10(N) - (a - b*mags))**2 ) / (len(mags) - 1)

    if b != 0:
        deficit = np.max(mags) - a/b
    else:
        deficit = 0

    return [b, a, eta, deficit]

quakes_to_indicator_features_mp_calculate_data = {
    "allQuakes": None,
    "tvalues": None
}
def quakes_to_indicator_features_mp_calculate(i):
    allQuakes = quakes_to_indicator_features_mp_calculate_data["allQuakes"]
    tvalues = quakes_to_indicator_features_mp_calculate_data["tvalues"]

    quakes = allQuakes[i]
    feats = []

    feats.append(len(quakes))
    feats.append(np.log(len(quakes) + 0.2))

    for t in tvalues:
        feats.append(TValue(quakes, t))

    feats.append(meanMag(quakes))
    feats.append(energyRate(quakes, 15))
    feats.extend(grLaw(quakes))

    return feats

def quakes_to_indicator_features(allQuakes, windowSize, nThreads, tvalues=[2.5, 3, 3.5, 4, 4.5, 5, 5.5]):
    allFeatures = []
    colNames = []

    colNames = [ name.format(windowSize) for name in [
            '{}-len-quakes',
            '{}-log-len-quakes',
    ]]
    colNames = colNames + [ '{}-tvalue-{:.1f}'.format(windowSize, t) for t in tvalues ]
    colNames = colNames + [ name.format(windowSize) for name in [
            '{}-mean-mag',
            '{}-energy-rate',
            '{}-gr-law-b',
            '{}-gr-law-a',
            '{}-gr-law-eta',
            '{}-gr-law-deficit'
        ]]

    quakes_to_indicator_features_mp_calculate_data["allQuakes"] = allQuakes
    quakes_to_indicator_features_mp_calculate_data["tvalues"] = tvalues

    allArgs = range(len(allQuakes))
    with mp.Pool(nThreads) as p:
        allFeatures = list(tqdm.tqdm(p.imap(quakes_to_indicator_features_mp_calculate, allArgs, chunksize=250), total=len(allArgs), smoothing=0.1))

    df = pd.DataFrame(allFeatures, columns=colNames)
    return df

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

        args_hash = hash((tuple(data["magnitude"]), tuple(inputw), tuple(outputw))) # A simple hash of the constructor args
        cache_filename = path.join(MEMOIZATION_DIR, f"eqtimewindows_cache_{args_hash}.pkl")

        if path.exists(cache_filename):
            print(f"Loading cached EQTimeWindows object from {cache_filename}...")
            with open(cache_filename, 'rb') as f:
                eqtw = pickle.load(f)

            self.inputw = eqtw.inputw
            self.outputw = eqtw.outputw
            self.data = eqtw.data
            self.nthreads = eqtw.nthreads

            self.x_quakes  = eqtw.x_quakes
            self.y_quakes  = eqtw.y_quakes

            self.x_quakes = eqtw.x_quakes
            self.y_quakes = eqtw.y_quakes

            self.x_stats = eqtw.x_stats
            self.y_stats = eqtw.y_stats

            self.x_indicators = eqtw.x_indicators
            self.x_indicators_joint = eqtw.x_indicators_joint

            return
            
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

        self.x_indicators = None
        self.x_indicators_joint = None

        # 4. Save the newly created object to the cache
        print(f"Saving this EQTimeWindows object to cache {cache_filename}...")
        with open(cache_filename, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def getXQuakesMaxMag(self):
        return [
            [ timewindow[:,MAG_INDEX].max() if len(timewindow) > 0 else 0 for timewindow in quakes ]
            for quakes in self.x_quakes
        ]
    def getYQuakesMaxMag(self):
        return [
            [ timewindow[:,MAG_INDEX].max() if len(timewindow) > 0 else 0 for timewindow in quakes ]
            for quakes in self.y_quakes
        ]

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
    
    def getXIndicators(self):
        if self.x_indicators is None:
            raise Exception("Seismicity indicators have not been calculated yet.")

        return self.x_indicators

    def calculateXIndicators(self):
        self.x_indicators = []
        allTvalues = [2.5, 3, 3.5, 4, 4.5, 5, 5.5]

        for quakes, T in zip(self.x_quakes, self.inputw):
            args_hash = hash((tuple(self.data["magnitude"]), T, tuple(allTvalues)))
            cache_filename = path.join(MEMOIZATION_DIR, f"indicators_cache_{args_hash}.pkl")

            if path.exists(cache_filename):
                print(f"Loading cached x_indicators object from {cache_filename}...")
                with open(cache_filename, 'rb') as f:
                    indic = pickle.load(f)
            else:
                indic = quakes_to_indicator_features(quakes, T, self.nthreads, tvalues=allTvalues)

                print(f"Saving this x_indicators object to cache {cache_filename}...")
                with open(cache_filename, 'wb') as f:
                    pickle.dump(indic, f, protocol=pickle.HIGHEST_PROTOCOL)

            self.x_indicators.append(indic)

        # Now we clean the indicators
        for indic in self.x_indicators:
            for col in indic.columns:
                values = indic[col].to_numpy()
                nanCount = len(values) - sum(np.isfinite(values))
                if nanCount > 0 and "tvalue" not in col:
                    indic = indic.drop(columns=[col])
                    print("Removing column due to NaNs:", col)
                elif nanCount > 1000: # Too many nans, remove
                    indic = indic.drop(columns=[col])
                    print("Removing column due to more than 1000 NaNs:", col)
                else: # Tvalue with not that much NaN, change to average
                    values[~np.isfinite(values)] = np.mean(values[np.isfinite(values)])
                    indic[col] = values
                # if "len-quakes" in col: # This catches {}-len-quakes and {}-log-len-quakes
                # if "mean-mag" in col:
                # if "energy-rate" in col:
                # if "gr-law-b" in col:
                # if "gr-law-a" in col:
                # if "gr-law-eta" in col:
                # if "gr-law-deficit" in col:

        # Finally, normalize
        for indic in self.x_indicators:
            for col in indic.columns:
                values = indic[col].to_numpy()
                indic[col] = (values - np.mean(values)) / np.std(values)

        self.x_indicators_joint = pd.concat(self.x_indicators, axis=1)

if __name__ == "__main__":
    data = load_dataset("ja")

    eqtw = EQTimeWindows(data, [7,15,30], 1, nthreads = 22)

    eqtw.calculateXIndicators()