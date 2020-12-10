from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from data.dataLoder import DataLoder


DATAPATH = "data/gbm/oil_field_data_for_gbm.xlsx"


def normalize(X):
    return (X - X.mean()) / X.std()


class GBMDataLoader(DataLoder):
    def __init__(self, datapath=DATAPATH):
        data = pd.read_excel(io=datapath)
        # drop first column
        data = data.drop(data.columns.values[0], axis=1)
        # split train data and target
        X, y = data.iloc[:, :-1], data.iloc[:, -1]
        self.trainX, self.testX = train_test_split(X, test_size=0.1, random_state=7)
        self.trainY, self.testY = train_test_split(y, test_size=0.1, random_state=7)

    def loadTrainData(self):
        return self.trainX, self.trainY

    def loadTestData(self):
        return self.testX, self.testY


if __name__ == "__main__":

    datapath = "data/gbm/oil_field_data_for_gbm.xlsx"
    data = pd.read_excel(io=datapath)
    # delete first column, shape(18, 11)
    data = data.drop(data.columns.valurs[0], axis=1)

    """
    # split train data and target
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    # print(X.shape, y.shape)  # (18, 10) (18,)
    # normX = (X - X.mean()) / X.std()
    # print(normX.head())
    trainX, testX = train_test_split(X, test_size=0.1, random_state=7)
    trainY, testY = train_test_split(y, test_size=0.1, random_state=7)
    # print(trainX.shape, testX.shape)
    # print(trainY.shape, testY.shape)
    """
