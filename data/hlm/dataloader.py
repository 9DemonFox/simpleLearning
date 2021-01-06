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
        x, y = self.__loadErosionData(datapath=EROSION_DATAPATH)
        w = self.__loadSoilData(datapath=SOIL_DATAPATH)
        self.trainX, self.trainY, self.trainW = x, y, w
        self.testX, self.testY, self.testW = x, y, w

    def __loadErosionData(self, datapath):
        erosion_data = pd.read_excel(io=datapath, index_col=0)
        y = erosion_data.values[:, 0].astype(float)
        x = erosion_data.values[:, 1:-1].astype(float)
        return x, y

    def __loadSoilData(self, datapath):
        soil_data = pd.read_excel(io=datapath, index_col=0)
        w = soil_data.values[:, :-1].astype(float)
        return w

    def loadTrainData(self, **kwargs):
        if "erosion_datapath" in kwargs.keys() and "soil_datapath" in kwargs.keys():
            trainX, trainY = self.__loadErosionData(datapath=kwargs.get("erosion_datapath"))
            trainW = self.__loadSoilData(datapath=kwargs.get("soil_datapath"))
            return trainW, trainX, trainY
        else:
            return self.trainW, self.trainX, self.trainY

    def loadTestData(self, **kwargs):
        if "erosion_datapath" in kwargs.keys() and "soil_datapath" in kwargs.keys():
            testX, testY = self.__loadErosionData(datapath=kwargs.get("erosion_datapath"))
            testW = self.__loadSoilData(datapath=kwargs.get("soil_datapath"))
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
    print(trainW, "\n", trainX, "\n", trainY)
    # print(testW, "\n", testX, "\n", testY)
    X = np.c_[np.ones(trainX.shape[0]), trainX]
    w = np.hstack((np.ones((trainW.shape[0], 1)), trainW))
    W = np.vstack((np.hstack((w, np.zeros((1, w.shape[1])))), np.hstack((np.zeros((1, w.shape[1])), w))))
    # print(W)
