import argparse
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection as ms
from tqdm import tqdm

def model_assessment(xTrain, xTest):
    scaler = preprocessing.StandardScaler()
    xTrain = scaler.fit_transform(xTrain)
    xTest = scaler.trasnform(xTest)
    return xTrain, xTest

def process():
    exec(open("./load_articles.py").read())
    exec(open("./form_dictionary.py").read())
    exec(open("./form_keywords.py").read())
    exec(open("./stock_parse.py").read())
    exec(open("./value_keywords.py").read())
    
    # transform to csv and obtain xTrain, yTrain, xTest, yTest
    # Turning json data to csv data
    xTrain, yTrain, xTest, yTest = 1,2,3,4

    xTrain, xTest = model_assessment(xTrain, xTest)

    print("Preprocessing Done")
    return xTrain, yTrain, xTest, yTest




