import numpy as np
import pandas as pd
import datetime as dt
import time

VALID_REGIONS = ["ja", "gr", "nz", "jma", "toho", "well", "stil", "jmatoho"]

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
        "jma": "../jma-japan-earthquakes-mag2.csv",
        "jmatoho": "../jma-japan-earthquakes-mag2.csv"
    }

    data = pd.read_csv(regionToDatafile[region])

    if minmag == 0 and (region == "jma" or region == "jmatoho"):
        minmag = 2.7

    data = data[data["magnitude"] >= minmag]
    data = data.query("year <= 2021").query("year < 2021 or month < 9")
    data = data.copy().reset_index(drop=True) # Ensures we are not working with a pandas slice

    data = add_time_columns(data)

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