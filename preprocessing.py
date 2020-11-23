import argparse
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection as ms
from tqdm import tqdm

def normalization(xTrain, xTest):
    scaler = preprocessing.StandardScaler()
    xTrain = scaler.fit_transform(xTrain)
    xTest = scaler.trasnform(xTest)
    return xTrain, xTest

# Unedited, can you unnormalized data
def pearson_graph(dfx, dfy):
    df = dfx.copy(deep=True)
    df['target'] = pd.Series(dfy['label'])
    correlation_matrix = df.corr(method='pearson')
    # sns.heatmap(correlation_matrix, annot=True)
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(correlation_matrix)
    plt.show()
    return None

def json_to_df():
    
    return x, y

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
    """
    
    # transform to csv and obtain xTrain, yTrain, xTest, yTest
    # Turning json data to csv data in either numpy format or df format
    x, y = json_to_df()
    
    # dataset spltting (cross validation)
    xTrain, yTrain, xTest, yTest = 1,2,3,4


    xTrain, xTest = normalization(xTrain, xTest)

    # Write to csv files
    xTrain.to_csv("xTrain", index=False)
    yTrain.to_csv("yTrain", index=False)
    xTest.to_csv("xTest", index=False)
    yTest.to_csv("yTest", index=False)

    print("Preprocessing Done")
    return None

# testing purposes
"""
def main():
    xTrain, yTrain, xTest, yTest = process()
    pearson_graph()

if __name__ == '__main__':
    main()
"""
    




