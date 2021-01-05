from sklearn.model_selection import train_test_split
import pandas as pd

from data.dataLoader import DataLoader

DATAPATH = "data/gbm/oil_field_data_for_gbm.xlsx"


def normalize(X):
    return (X - X.mean()) / X.std()


class GBMDataLoader(DataLoader):
    def __init__(self, datapath=DATAPATH):
        # split train data and target
        X, y = self.__loadExcelData(datapath=datapath)
        self.trainX, self.testX = train_test_split(X, test_size=0.1, random_state=7)
        self.trainY, self.testY = train_test_split(y, test_size=0.1, random_state=7)

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
        if "train_path" in kwargs.keys():
            train_datapath = kwargs["train_path"]
            trainX, trainY = self.__loadExcelData(datapath=train_datapath)
            return trainX, trainY
        else:
            return self.trainX, self.trainY

    def loadTestData(self, **kwargs):
        if "test_path" in kwargs.keys():
            test_datapath = kwargs["test_path"]
            testX, testY = self.__loadExcelData(datapath=test_datapath)
            return testX, testY
        else:
            return self.testX, self.testY


if __name__ == "__main__":
    datapath = "data/gbm/oil_field_data_for_gbm.xlsx"

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
