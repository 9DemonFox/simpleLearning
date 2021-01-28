from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from data.dataLoader import DataLoader


DATAPATH = "test.xlsx"

def fra_Data(data, fra):  # fra为正整数
    ret = data.copy()
    while fra > 1:
        ret = np.concatenate((ret, data), axis=0)
        fra -= 1
    return ret

class IBRTDataLoader(DataLoader):
    def __init__(self):
        pass

    def __loadExcelData(self, datapath):
        """
        :param data_path: excel数据 第1列为Y
        :return:
        """
        df = pd.read_excel(datapath, index_col=0)
        y = df.values[:, 0]
        X = df.values[:, 1:]
        return X, y

    def loadTrainData(self, **kwargs):
        assert "train_path" in kwargs.keys()
        train_datapath = kwargs["train_path"]
        trainX, trainY = self.__loadExcelData(datapath=train_datapath)
        return trainX, trainY

    def loadTestData(self, **kwargs):
        assert "test_path" in kwargs.keys()
        test_datapath = kwargs["test_path"]
        testX, testY = self.__loadExcelData(datapath=test_datapath)
        return testX, testY

    def loadPredictData(self, **kwargs):
        assert "predict_path" in kwargs.keys()
        predict_path = kwargs["predict_path"]
        df = pd.read_excel(io=predict_path, index_col=0)
        predictX = df.values[:, :]
        return predictX

if __name__ == "__main__":
    train_datapath = "ibrt_train_data.xlsx"
    test_datapath = "ibrt_test_data.xlsx"
    predict_datapath = "ibrt_predict_data.xlsx"
    ibrt_dataloader = IBRTDataLoader()
    trainX, trainY = ibrt_dataloader.loadTrainData(train_path=train_datapath)
    testX, testY = ibrt_dataloader.loadTestData(test_path=test_datapath)
    predictX = ibrt_dataloader.loadPredictData(predict_path=predict_datapath)
    print(trainX.shape, trainY.shape)
    print(testX.shape, testY.shape)
    print(predictX.shape)