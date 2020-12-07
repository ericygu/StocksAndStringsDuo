import numpy as np
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm


def normalization(xTrain, xTest):
    std_scale = preprocessing.StandardScaler(with_mean=False)
    cols = list(xTrain.columns)
    xTrain = std_scale.fit_transform(xTrain)
    xTest = std_scale.transform(xTest)
    xTrain = pd.DataFrame(xTrain, columns=cols)
    xTest = pd.DataFrame(xTest, columns=cols)
    return xTrain, xTest


def extract_features(df):
    df['datetime'] = pd.to_datetime(df['datetime'], format="%Y-%m-%d %H:%M:%S")
    df.insert(loc=0, column='dt_year', value=df['datetime'].dt.year)
    df.insert(loc=0, column='dt_month', value=df['datetime'].dt.month)
    df.insert(loc=0, column='dt_day', value=df['datetime'].dt.day)
    df.insert(loc=0, column='dt_dayofweek', value=df['datetime'].dt.dayofweek)
    df.insert(loc=0, column='dt_hour', value=df['datetime'].dt.hour)
    df = df.drop(columns=['datetime'])
    return df


# use pearson correlation to remove redundant features and features with nan correlation to stock price change
def pearson_graph(dfx, dfy):
    matrix = dfx.to_numpy()
    matrix = np.hstack((dfy['stock_change'].to_numpy()[:,np.newaxis], matrix))
    matrix = matrix.transpose()
    #valid = ~np.isnan(matrix).any(axis=0)
    corr = np.ma.corrcoef(matrix)
    corr = np.ma.getdata(corr)
    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        if pd.isnull(corr[0,i]):
            if columns[i]:
                columns[i] = False
        for j in range(i+1, corr.shape[0]):
            if corr[i,j] >= 0.9:
                if columns[j]:
                    columns[j] = False
    selected_columns = dfx.columns[columns[1:]]
    return selected_columns


def get_csv():
    x = pd.read_csv('X.csv')
    y = pd.read_csv('Y.csv')
    # extract features of datetime column
    x = extract_features(x)
    # drop datetime column from y data
    y = y.drop(columns=['datetime'])
    return x, y


def update_data():
    # Acquire dataset Don't run this again, it generates new data and replaces what we have. We will have to write
    # something that adds to current data later. I took care of reading from the current data files and converting
    # them to dataframes/csv. -Nathan
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
    return None


def process(xTrain, yTrain, xTest, yTest):
    # Pearson graph of features with datetime extracted (can be commented out to not show pearson correlation graph)
    selected_columns = pearson_graph(xTrain, yTrain)
    # corr.to_csv("corr_matrix.csv")

    # normalize the x data
    xTrain, xTest = normalization(xTrain, xTest)

    # remove reduntant features
    xTrain_pearson = xTrain[selected_columns[1:]]
    xTest_pearson = xTest[selected_columns[1:]]

    # convert dataframes to csv files.
    """
    xTrain.to_csv("xTrain.csv", index=False)
    xTrain_pearson.to_csv("xTrain.csv", index=False)
    xTest.to_csv("xTest.csv", index=False)
    xTest_pearson.to_csv("xTest.csv", index=False)
    """
    return xTrain_pearson, yTrain, xTest_pearson, yTest


"""
# testing purposes
def main():
    process()

if __name__ == '__main__':
    main()
"""
