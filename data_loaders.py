import numpy as np
import pandas as pd
import datetime as dt
import time

def add_time_columns(data):
    initDate = dt.datetime(2000, 1, 1)
    newCol = []
    newCol2 = []

    data = data.copy() # Just to avoid changing the callee data

    for i in range(len(data)):
        datum = data.iloc[i,:]
        dateObj = dt.datetime(datum["year"], datum["month"], datum["day"]) # Assume existence of "year", "month" and "day"
        ndays = (dateObj - initDate).days

        try:
            dateObj = dt.datetime(datum["year"], datum["month"], datum["day"], int(datum["hour"]), int(datum["minute"]), int(datum["second"]))
        except Exception:
            dateObj = dt.datetime(datum["year"], datum["month"], datum["day"], int(datum["hour"]), int(datum["minute"]))

        nsecs = dateObj.timestamp() - initDate.timestamp()
        
        newCol.append(ndays)
        newCol2.append(nsecs)

    data["day.number"] = newCol
    data["time.seconds"] = newCol2

    return data

valid_regions = ["ja", "gr", "nz", "jma", "toho", "well", "stil", "jmatoho"]
def load_dataset(region, minmag=0):
    if region not in valid_regions:
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

    data = data[data["magnitude"] >= minmag]
    data = data.query("year <= 2021").query("year < 2021 or month < 9")
    data = data.copy().reset_index(drop=True) # Ensures we are not working with a pandas slice

    data = add_time_columns(data)

    if region in ["toho", "well", "stil", "jmatoho"]:
        raise Exception("Region not implemented")
    
    return data