import argparse
import datetime
import pandas as pd
import numpy as np
from load_articles import read_articles
from get_keywords import read_keywords
from stock_parse import read_hourlyStock


def json_to_df(nkeywords):
    def get_datetime(article):
        return datetime.datetime.strptime(article['date_published'], "%Y-%m-%dT%H:%M:%SZ")

    def get_dtnearest_hr(article):
        return (get_datetime(article) - datetime.timedelta(hours=5)).replace(second=0, minute=0)

    def get_timeframe(data):
        dt_start = get_dtnearest_hr(data[0])
        dt_end = get_dtnearest_hr(data[-1])
        diff = dt_end - dt_start
        tot_hours = diff.days * 24 + diff.seconds / 3600 + 1
        timeframe = pd.date_range(start=str(dt_start), end=str(dt_end), periods=tot_hours)
        return timeframe

    data = read_articles()
    keywords = read_keywords()[0:nkeywords]
    hourlyStock = read_hourlyStock()
    data = sorted(data, key=lambda entry: get_datetime(entry))
    timeframe = get_timeframe(data)
    feat_mat = np.zeros((len(timeframe), len(keywords)), int)
    labels = np.array(list(hourlyStock.values()))

    j = 0
    for i in range(len(timeframe)):
        if j < len(data) and timeframe[i] == get_dtnearest_hr(data[j]):
            # labels[i] = data[j]['delta']
            while j < len(data) and timeframe[i] == get_dtnearest_hr(data[j]):
                feat_row = [1 if keyword in data[j]['keywords'] else 0 for keyword in keywords]
                feat_mat[i] += feat_row
                j += 1
        else:
            feat_mat[i] = feat_mat[i - 1]

    X = pd.DataFrame(feat_mat, timeframe, keywords)
    y = pd.DataFrame(labels, timeframe, ['stock_change'])
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("x_name", help="name for feature matrix csv file (remember to include .csv)")
    parser.add_argument("y_name", help="name for labels csv file (remember to include .csv)")
    parser.add_argument("--nkeywords", default=5194, type=int,
                        help="top n number of keywords to use, default=5194 includes all keywords")
    args = parser.parse_args()

    X, y = json_to_df(args.nkeywords)
    X.to_csv(args.x_name)
    y.to_csv(args.y_name)


if __name__ == "__main__":
    main()
