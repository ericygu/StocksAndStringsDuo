# -*- coding: utf-8 -*-

# relies on JSON input of the form
# {title -> string, author -> string, time -> datetime, text -> string}
import json  # will use
import datetime
import requests
from load_articles import read_articles, write_articles


def updateJSON_prices(sym):

    APIKey = "442ONKXSVHA79170"  # redact in submissions
    APIbase = "https://www.alphavantage.co/query"

    def rqstStockTSDataDaily(sym):
        r = None
        tries = 0
        maxTries = 10
        while not r and tries < maxTries:
            r = requests.get(APIbase, params={
                "function": "TIME_SERIES_DAILY", "symbol": sym, "outputsize": "full", "apikey": APIKey})
            tries += 1

        if not r:
            raise ValueError("Something unexpected happened.")

        return r.json()["Time Series (Daily)"]

    def getTSDataDaily(stockDat, t):
        try:
            e = stockDat[t.strftime("%Y-%m-%d")]
            return {"open": round(float(e["1. open"]), 2),
                    "close": round(float(e["4. close"]), 2),
                    "high": round(float(e["2. high"]), 2),
                    "low": round(float(e["3. low"]), 2),
                    "volume": int(e["5. volume"])}
        except KeyError:
            return False

    def getTSDataDailyForceSuccessFuture(stockDat, t, maxFail):
        y = getTSDataDaily(stockDat, t)
        i = 0
        while not y:
            t += datetime.timedelta(days=1)
            y = getTSDataDaily(stockDat, t)
            i += 1
            if i == maxFail:
                return False

        return (t, y)

    dailyDat = rqstStockTSDataDaily(sym)
    dat = read_articles()
    for article in dat:
        # round down to nearest hour
        t = (datetime.datetime.strptime(
            article["date_published"], "%Y-%m-%dT%H:%M:%SZ") - datetime.timedelta(hours=5)).replace(second=0, minute=0)
        #            print(dailyDat)
        result = getTSDataDailyForceSuccessFuture(dailyDat, t, 1000)
        if not result:
            article["delta"] = float(0)
        else:
            article["delta"] = result[1]["close"] - result[1]["open"]
            if t.date() == result[0].date():
                article["const"] = True
            else:
                article["const"] = False

    write_articles(dat)


###################################TEST CODE HERE###################################
# p = rqstStockTSData("FB", 30)       #check
# q = getTSData("FB", 30, datetime.datetime(2020, 1, 22, 16, 0))      #check
# use average of opening and closing price
# with open(dataFile, 'r') as f:
#    dat = json.load(f)
#    dailyDat = rqstStockTSDataDaily("FB")
#    for article in dat:
#        t = (datetime.datetime.strptime(article["date_published"], "%Y-%m-%dT%H:%M:%SZ") - datetime.timedelta(hours=5)).replace(second = 0, minute = 0)
# print(t)
#        t_actual, s = getTSDataDailyForceSuccessFuture(dailyDat, t, 1000)
# print(s)
#        print("Time: " + str(t) + " | Actual indexed time: " + str(t_actual) + " | Stock difference: %0.2f" % (s["close"] - s["open"]))
if __name__ == '__main__':
    updateJSON_prices("FB")
