import numpy as np
import pandas as pd
import sys, os

from data.dataLoader import DataLoader

DIR = os.path.dirname(os.path.abspath(__file__))

EROSION_DATAPATH = DIR + "/train_erosion_data.xlsx"
SOIL_DATAPATH = DIR + "/soil_data.xlsx"


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
        # x, y = self.__loadErosionData(datapath=EROSION_DATAPATH)
        # w = self.__loadSoilData(datapath=SOIL_DATAPATH)
        # self.trainX, self.trainY, self.trainW = x, y, w
        # self.testX, self.testY, self.testW = x, y, w
        pass

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
        assert "train_path" in kwargs.keys()
        assert "soil_path" in kwargs.keys()
        train_path, soil_path = kwargs.get("train_path"), kwargs.get("soil_path")
        trainX, trainY = self.__loadErosionData(datapath=train_path)
        trainW = self.__loadSoilData(datapath=soil_path)
        return trainW, trainX, trainY

    def loadTestData(self, **kwargs):
        assert "test_path" in kwargs.keys()
        assert "soil_path" in kwargs.keys()
        test_path, soil_path = kwargs.get("test_path"), kwargs.get("soil_path")
        testX, testY = self.__loadErosionData(datapath=test_path)
        testW = self.__loadSoilData(datapath=soil_path)
        return testW, testX, testY

    def loadPredictData(self, **kwargs):
        assert "predict_path" in kwargs.keys()
        assert "soil_path" in kwargs.keys()
        predict_path, soil_path = kwargs.get("predict_path"), kwargs.get("soil_path")
        df = pd.read_excel(io=predict_path, index_col=0)
        predictX = df.values[:, :-1].astype(float)
        predictW = self.__loadSoilData(datapath=soil_path)
        return predictW, predictX


if __name__ == "__main__":
    dataloader = HLMDataLoader()

    soil_path = "soil_data.xlsx"
    train_path = "train_erosion_data.xlsx"
    test_path = "test_erosion_data.xlsx"
    predict_path = "predict_erosion_data.xlsx"

    trainW, trainX, trainY = dataloader.loadTrainData(train_path=train_path, soil_path=soil_path)
    print(trainW.shape, trainX.shape, trainY.shape)

    testW, testX, testY = dataloader.loadTestData(test_path=test_path, soil_path=soil_path)
    print(testW.shape, testX.shape, testY.shape)

    predictW, predictX = dataloader.loadPredictData(predict_path=predict_path, soil_path=soil_path)
    print(predictW.shape, predictX.shape)
    # X = np.c_[np.ones(trainX.shape[0]), trainX]
    # w = np.hstack((np.ones((trainW.shape[0], 1)), trainW))
    # W = np.vstack((np.hstack((w, np.zeros((1, w.shape[1])))), np.hstack((np.zeros((1, w.shape[1])), w))))
    # print(W)
