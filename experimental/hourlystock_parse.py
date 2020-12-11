import json
import datetime
import requests
import pandas as pd
from load_articles import read_articles, write_articles


def write_hourlyStock(hourlyStock):
    with open('hourlyStock.json', 'w') as fp:
        json.dump(hourlyStock, fp)


def read_hourlyStock():
    with open('hourlyStock.json') as f:
        return json.load(f)


def updateJSON_prices(sym):
    APIKey = "442ONKXSVHA79170"  # redact in submissions
    APIbase = "https://www.alphavantage.co/query"

    def rqstStockTSIntraDay(sym):
        r = None
        tries = 0
        maxTries = 10
        while not r and tries < maxTries:
            r = requests.get(APIbase, params={
                "function": "TIME_SERIES_INTRADAY", "symbol": sym, "interval": "60min", "outputsize": "full", "apikey": APIKey})
            tries += 1

        if not r:
            raise ValueError("Something unexpected happened.")

        return r.json()["Time Series (IntraDay)"]

    def getTSDataIntraDay(stockDat, t):
        try:
            e = stockDat[t.strftime("%Y-%m-%d %H:%M:%S")]
            return {"open": round(float(e["1. open"]), 2),
                    "close": round(float(e["4. close"]), 2),
                    "high": round(float(e["2. high"]), 2),
                    "low": round(float(e["3. low"]), 2),
                    "volume": int(e["5. volume"])}
        except KeyError:
            return False

    def getTSDataDailyForceSuccessFuture(stockDat, t, maxFail):
        y = getTSDataIntraDay(stockDat, t)
        i = 0
        while not y:
            t += datetime.timedelta(days=1)
            y = getTSDataIntraDay(stockDat, t)
            i += 1
            if i == maxFail:
                return False

        return (t, y)

    def get_datetime(data, i):
        return datetime.datetime.strptime(data['datetime'][i], "%Y-%m-%d %H:%M:%S")

    def get_timeframe(data):
        dt_start = get_datetime(data, 0)
        dt_end = get_datetime(data, len(data)-1)
        diff = dt_end - dt_start
        tot_hours = diff.days * 24 + diff.seconds / 3600 + 1
        timeframe = pd.date_range(start=str(dt_start), end=str(dt_end), periods=tot_hours)
        return timeframe

    # Updates hourlyStock
    intraDayDat = rqstStockTSIntraDay(sym)
    dat = pd.read_csv('X.csv')
    timeframe = get_timeframe(dat)
    hourlyStock = dict.fromkeys(timeframe)
    for t in timeframe:
        result = getTSDataDailyForceSuccessFuture(intraDayDat, t, 1000)
        hourlyStock[t] = result[1]["close"] - result[1]["open"]
    hourlyStock = {str(k): v for k, v in hourlyStock.items()}
    write_hourlyStock(hourlyStock)