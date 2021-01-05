import numpy as np
import pandas as pd
import sys, os

from data.dataLoader import DataLoader

DIR = os.path.dirname(os.path.abspath(__file__))

EROSION_DATAPATH = DIR + "/fake_erosion_data.xlsx"
SOIL_DATAPATH = DIR + "/fake_soil_data.xlsx"


def loadData(erosion_datapath, soil_datapath):
    erosion_data = pd.read_excel(io=erosion_datapath)
    soil_data = pd.read_excel(io=soil_datapath)

    # drop first column
    erosion_data = erosion_data.drop(erosion_data.columns.values[0], axis=1)
    soil_data = soil_data.drop(soil_data.columns.values[0], axis=1)

    x = np.array(erosion_data.iloc[:, :-1])  # assume shape is (N, m), default data is (4, 1)
    y = np.array(erosion_data.iloc[:, -1])  # assume shape is (N, )
    w = np.array(soil_data.iloc[:, :]).flatten()  # assume shape is (q, )
    return w, x, y


class HLMDataLoader(DataLoader):
    def __init__(self):
        erosion_data = pd.read_excel(io=EROSION_DATAPATH)
        soil_data = pd.read_excel(io=SOIL_DATAPATH)

        # drop first column
        erosion_data = erosion_data.drop(erosion_data.columns.values[0], axis=1)
        soil_data = soil_data.drop(soil_data.columns.values[0], axis=1)

        x = np.array(erosion_data.iloc[:, :-1])
        y = np.array(erosion_data.iloc[:, -1])
        w = np.array(soil_data.iloc[:, :]).flatten()
        self.trainX, self.trainY, self.trainW = x, y, w
        self.testX, self.testY, self.testW = x, y, w

    def loadTrainData(self, **kwargs):
        if "erosion_datapath" in kwargs.keys() and "soil_datapath" in kwargs.keys():
            trainW, trainX, trainY = loadData(kwargs["erosion_datapath"], kwargs["soil_datapath"])
            return trainW, trainX, trainY
        else:
            return self.trainW, self.trainX, self.trainY

    def loadTestData(self, **kwargs):
        if "erosion_datapath" in kwargs.keys() and "soil_datapath" in kwargs.keys():
            testW, testX, testY = loadData(kwargs["erosion_datapath"], kwargs["soil_datapath"])
            return testW, testX, testY
        else:
            return self.testW, self.testX, self.testY


if __name__ == "__main__":
    dataloader = HLMDataLoader()
    '''
    trainW, trainX, train = dataloader.loadTrainData(erosion_datapath="fake_erosion_data.xlsx",
                                                     soil_datapath="fake_soil_data.xlsx")
    '''
    trainW, trainX, trainY = dataloader.loadTrainData()
    '''
    testW, testX, testY = dataloader.loadTestData(erosion_datapath="fake_erosion_data.xlsx",
                                                  soil_datapath="fake_soil_data.xlsx")
    '''
    testW, testX, testY = dataloader.loadTestData()
    print(trainW.shape, '\n', trainX.shape, '\n', trainY.shape)
