import argparse
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection as ms
from form_csv import json_to_df
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


def normalization(xTrain, xTest):
    std_scale = preprocessing.StandardScaler()
    xTrain = std_scale.fit_transform(xTrain)
    xTest = std_scale.transform(xTest)
    return xTrain, xTest

def extract_features(df):
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df = df.drop(columns=['date'])
    return df

# no need for normalized data with pearson correlation graph
def pearson_graph(dfx, dfy):
    df = dfx.copy(deep=True)
    df['target'] = pd.Series(dfy['label'])
    correlation_matrix = df.corr(method='pearson')
    # sns.heatmap(correlation_matrix, annot=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(correlation_matrix)
    plt.show()
    return None


def process():
    # Acquire dataset
    # Don't run this again, it generates new data and replaces what we have. We will have to write something that adds to current data later.
    # I took care of reading from the current data files and converting them to dataframes/csv. -Nathan
    """
    exec(open("./load_articles.py").read())
    exec(open("./form_dictionary.py").read())
    exec(open("./form_keywords.py").read())
    exec(open("./stock_parse.py").read())
    exec(open("./value_keywords.py").read())

    # transform to csv and obtain xTrain, yTrain
    # Turning json data to csv data in either numpy format or df format
    exec(open("./form_csv.py").read())
    """
    # convert data from json to dataframe
    n = 5194  # number of keywords to include in the dataset
    x, y = json_to_df(n)

    # extract features of datetime column
    x = extract_features(x)
    # drop datetime column from y data
    y = y.drop(columns=['date'])

    # split data into training set and test set
    xTrain, xTest, yTrain, yTest = ms.train_test_split(x, y, test_size=0.2)

    """
    xTrain.to_csv("xTrain_orig", index=False)
    xTest.to_csv("xTest_orig", index=False)
    yTrain.to_csv("yTrain_orig", index=False)
    yTest.to_csv("yTest_orig", index=False)
    """

    # normalize the x data
    xTrain, xTest = normalization(xTrain, xTest)

    # Pearson graph of features with datetime extracted (can be commented out to not show pearson correlation graph)
    pearson_graph(xTrain, yTrain)

    # convert dataframes to csv files
    xTrain.to_csv("xTrain", index=False)
    xTest.to_csv("xTest", index=False)
    yTrain.to_csv("yTrain", index=False)
    yTest.to_csv("yTest", index=False)

    print("Preprocessing Done")
    return None


# testing purposes
"""
def main():
    process()

if __name__ == '__main__':
    main()
"""
