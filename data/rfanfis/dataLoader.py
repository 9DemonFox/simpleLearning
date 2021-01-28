from torch.utils.data import TensorDataset, DataLoader
import pandas
import torch
class anfisDataLoader(DataLoader):
    def __init__(self, num_cases=100, batch_size=16):
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
        assert "train_path" in kwargs.keys()
        trainX, trainY = self.__loadExcelData(kwargs.get("train_path"))
        trainX = torch.from_numpy(trainX)
        trainY = torch.from_numpy(trainY)
        train_db = TensorDataset(trainX, trainY)
        train_dl = DataLoader(train_db, batch_size=16, shuffle=True)
        trainX = train_dl.dataset.tensors[0].numpy().reshape(-1, 3)
        trainY = train_dl.dataset.tensors[1].numpy().reshape(-1, 1)
        return trainX, trainY

    def loadTestData(self, **kwargs):
        assert "test_path" in kwargs.keys()
        testX, testY = self.__loadExcelData(kwargs.get("test_path"))
        testX, testY = torch.from_numpy(testX), torch.from_numpy(testY)
        test_db = TensorDataset(testX, testY)
        test_dl = DataLoader(test_db, batch_size=16, shuffle=True)
        testX = test_dl.dataset.tensors[0].numpy().reshape(-1, 3)
        testY = test_dl.dataset.tensors[1].numpy().reshape(-1, 1)
        return testX, testY

if __name__ == "__main__":
    model = anfisDataLoader()
    print('shape: ', model.loadTrainData(train_path="RFANFIS_TRAIN_DATA.xlsx")[0].shape)
