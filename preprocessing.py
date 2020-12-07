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
    """
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['weekday'] = df['datetime'].dt.dayofweek
    df['hour'] = df['datetime'].dt.hour
    """
    df.insert(loc=0, column='dt_year', value=df['datetime'].dt.year)
    df.insert(loc=0, column='dt_month', value=df['datetime'].dt.month)
    df.insert(loc=0, column='dt_day', value=df['datetime'].dt.day)
    df.insert(loc=0, column='dt_dayofweek', value=df['datetime'].dt.dayofweek)
    df.insert(loc=0, column='dt_hour', value=df['datetime'].dt.hour)
    df = df.drop(columns=['datetime'])
    return df


# use pearson correlation to remove redundant features and features with nan correlation to stock price change
def pearson_graph(dfx, dfy):
    df = dfx.copy(deep=True)
    # df['target'] = dfy['stock_change']
    df.insert(loc=0, column='y_target', value=dfy['stock_change'])
    corr = df.corr(method='pearson')
    # sns.heatmap(correlation_matrix, annot=True)
    # fig, ax = plt.subplots(figsize=(10, 10))
    # sns.heatmap(correlation_matrix)
    # plt.show()
    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in tqdm(range(corr.shape[0])):
        if pd.isnull(corr.iloc[0, i]):
            if columns[i]:
                columns[i] = False
        else:
            for j in range(i + 1, corr.shape[0]):
                if corr.iloc[i, j] >= 0.9:
                    if columns[j]:
                        columns[j] = False
    selected_columns = df.columns[columns]
    return selected_columns, corr


def get_csv():
    # convert data from json to dataframe
    # n = 5194  # number of keywords to include in the dataset
    # x, y = json_to_df(n)
    x = pd.read_csv('X.csv')
    y = pd.read_csv('Y.csv')
    # extract features of datetime column
    x = extract_features(x)
    # drop datetime column from y data
    y = y.drop(columns=['datetime'])
    x.to_csv('X_new.csv', index=False)
    y.to_csv('Y_new.csv', index=False)
    return None


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
    selected_columns, corr = pearson_graph(xTrain, yTrain)
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

    print("Preprocessing Done")
    return xTrain_pearson, yTrain, xTest_pearson, yTest


"""
# testing purposes
def main():
    process()

if __name__ == '__main__':
    main()
"""
