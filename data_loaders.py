import numpy as np
import pandas as pd
import datetime as dt
import pickle
import time

VALID_REGIONS = ["ja", "gr", "nz", "jma", "toho", "well", "stil", "jmatoho"]
VALID_CLUSTER_REGIONS = ["ja", "nz"]

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

def load_cluster_dataset(region, minmag=0):
    if region not in VALID_CLUSTER_REGIONS:
        raise Exception(f"Invalid region: {region}.")

    regionToDatafile = {
        "ja": "../japan-clusterized-dataframes.pickle",
        "nz": "../newzealand-clusterized-dataframes.pickle"
    }

    data = pklload(regionToDatafile[region])

    for i in range(len(data)):
        d = data[i]
        d = d[d["magnitude"] >= minmag]
        d = d.query("year <= 2021").query("year < 2021 or month < 9")
        d = d.copy().reset_index(drop=True) # Ensures we are not working with a pandas slice
        data[i] = d
    
    return data

class EQTimeWindows:
    def __init__(self, data, inputw=7, outputw=1):
        try:
            for col in ["day.number", "magnitude", "latitude", "longitude", "depth", "time.seconds"]:
                data[col]
        except:
            raise Exception("Dataframe does not have the required columns.")
        
        self.inputw = inputw
        self.outputw = outputw
        self.data = data

        # prefix 'x' means the variable refers to the X space, the independent variable space (the previous time windows)
        # prefix 'y' means the dependent variable Y space, often the next-window features
        self.x_quakes  = []
        self.x_maxMag  = []
        self.x_meanMag = []
        self.x_logN    = []

        self.x_quakes_N    = []
        self.x_quakes_logN = []
        #allX_seismicity = []

        self.y_dayNumbers = []

        # Window size
        W = inputw

        # Prediction window
        PRED_WINDOW = outputw

        dayNumbers = data["day.number"] # To make things faster, save it in a variable
        maxDayNumber = (dt.datetime(2021, 8, 31) - dt.datetime(2000, 1, 1)).days
        # NOTE: day.number of 2000/01/01 is 0, so we do not need to add 1 to maxDayNumber

        for i in range(W-1, maxDayNumber + 1 - PRED_WINDOW):
            quakeWindow = data[ (dayNumbers > i - W) * (dayNumbers <= i) ]
            predWindow  = data[ (dayNumbers > i) * (dayNumbers <= i+PRED_WINDOW) ]
            self.y_dayNumbers.append(i + 1)

            if len(quakeWindow) > 0:
                quakeSequence = np.array(quakeWindow[["time.seconds", "magnitude", "longitude", "latitude", "depth"]])
                quakeSequence[:,0] = quakeSequence[:,0] - (i-W+1) * 24 * 60 * 60
            else:
                quakeSequence = np.array([])

            if len(predWindow) > 0:
                self.x_maxMag.append(predWindow["magnitude"].max())
                self.x_meanMag.append(predWindow["magnitude"].mean())
            else:
                self.x_maxMag.append(0)
                self.x_meanMag.append(0)

            self.x_quakes.append(quakeSequence)
            self.x_logN.append(np.log(len(predWindow) + 1)) #variable to predict

            self.x_quakes_N.append(len(quakeSequence))
            self.x_quakes_logN.append(np.log(len(quakeSequence) + 1))

    def getBaselineStds(self, tlambda):
        df = self.data
        slic = df[df["year"] < 2011]

        timeStd      = np.std(np.diff(slic["time.seconds"])) * args.tlambda
        magnitudeStd = df["magnitude"].std()
        depthStd     = df["depth"].std()
        latitudeStd  = df["latitude"].std()
        longitudeStd = df["longitude"].std()

        return [timeStd, magnitudeStd, longitudeStd, latitudeStd, depthStd]