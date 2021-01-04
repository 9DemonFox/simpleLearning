import pandas

from data.dataLoader import DataLoader


class REBETDataLoader(DataLoader):
    def __init__(self):
        pass
    
    def __loadExcelData(self, data_path):
        """
        :param data_path: excel数据 第1列为Y
        :return:
        """
        df = pandas.read_excel(data_path, index_col=0)
        y = df.values[:, 0]
        x = df.values[:, 1:]
        return x, y
    
    def loadTrainData(self, **kwargs):
        trainX, trainY = self.__loadExcelData(kwargs.get("train_path"))
        return trainX, trainY

    def loadTestData(self, **kwargs):
        testX, testY = self.__loadExcelData(kwargs.get("test_path"))
        return testX, testY
    
    def loadPredictData(self, **kwargs):
        """
        :param predict_path 预测的输入
        :return: 加载数据
        """
        data_path = kwargs.get("predict_path")
        df = pandas.read_excel(data_path, index_col=0)
        x = df.values[:, 0:]
        return x


if __name__ == "__main__":
    datapath1 = "./data_train.xlsx"
    datapath2 = "./data_test.xlsx"
    dataloder = REBETDataLoader(datapath1, datapath2)
    trainX, trainY = dataloder.loadTrainData(train_path=datapath1)
    testX, testY = dataloder.loadTestData(test_path=datapath2)
