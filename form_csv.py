import argparse
import json
import datetime
import pandas as pd
import numpy as np
from load_articles import read_articles
from get_keywords import read_keywords

def json_to_df(nkeywords):

	def get_datetime(article):
		return datetime.datetime.strptime(article['date_published'], "%Y-%m-%dT%H:%M:%SZ")

	def get_dtnearest_hr(article):
		return (get_datetime(article) - datetime.timedelta(hours=5)).replace(second=0, minute=0)

	data = read_articles()
	data = sorted(data, key = lambda entry:get_datetime(entry))
	keywords = read_keywords()[0:nkeywords]
	df_rows = []
	labels = []
	dt_hr = []
	count = 0

	for i in range(len(data)):
		detected = [1 if keyword in data[i]['keywords'] else 0 for keyword in keywords]
		if i>0 and get_dtnearest_hr(data[i])==get_dtnearest_hr(data[i-1]):
			df_rows[count-1] += detected
		else:
			df_rows.append(detected)
			labels.append(data[i]['delta'])
			dt_hr.append(get_dtnearest_hr(data[i]).strftime("%Y-%m-%d hr%H"))
			count += 1
	df_rows = [dict(zip(keywords, entry)) for entry in df_rows]

	X = pd.DataFrame(df_rows)
	X.index = dt_hr
	y = pd.DataFrame({'fbstock_change':labels})
	y.index = dt_hr
	return X, y
    
"""
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("x_name", help="name for feature matrix csv file (remember to include .csv)")
	parser.add_argument("y_name", help="name for labels csv file (remember to include .csv)")
	parser.add_argument("--nkeywords", default=5194, type=int, help="top n number of keywords to use, default=5194 includes all keywords")
	args = parser.parse_args()

	X, y = json_to_df(args.nkeywords)
	X.to_csv(args.x_name)
	y.to_csv(args.y_name)


if __name__ == "__main__":
    main()
"""
