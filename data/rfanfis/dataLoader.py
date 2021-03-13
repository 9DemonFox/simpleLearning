from torch.utils.data import TensorDataset, DataLoader
import pandas
import torch
class ANFISDataLoader(DataLoader):
    def __init__(self, num_cases=100, batch_size=16):
        pass

    def __loadExcelData(self, data_path):
        """
        :param data_path: excel数据 第1列为Y
        :return:
        """
        df = pandas.read_excel(data_path, index_col=0)
        #print(df)
        y = df.values[:, 0]
        x = df.values[:, 1:]
        return x, y

    def loadTrainData(self, **kwargs):
        assert "train_path" in kwargs.keys()
        trainX, trainY = self.__loadExcelData(data_path=kwargs.get("train_path"))

        trainX = torch.from_numpy(trainX)
        trainY = torch.from_numpy(trainY)
        #print('trainX',trainX)
        #print('trainY', trainY)
        train_db = TensorDataset(trainX, trainY)
        train_dl = DataLoader(train_db, batch_size=16, shuffle=False)
        trainX = train_dl.dataset.tensors[0].numpy()
        trainY = train_dl.dataset.tensors[1].numpy()
        return trainX, trainY

    def loadTestData(self, **kwargs):
        assert "test_path" in kwargs.keys()
        testX, testY = self.__loadExcelData(data_path=kwargs.get("test_path"))
        testX, testY = torch.from_numpy(testX), torch.from_numpy(testY)
        test_db = TensorDataset(testX, testY)
        test_dl = DataLoader(test_db, batch_size=16, shuffle=False)
        testX = test_dl.dataset.tensors[0].numpy()
        testY = test_dl.dataset.tensors[1].numpy()
        return testX, testY

    def loadPredictData(self, **kwargs):
        assert "predict_path" in kwargs.keys()

        predictX = pandas.read_excel(kwargs.get("predict_path"), index_col=0).values[:,:]
        # print(predictX)
        predicty = pandas.read_excel(kwargs.get("predict_path"), index_col=0).values[:,0]
        predictX, predicty = torch.from_numpy(predictX), torch.from_numpy(predicty)
        predict_db = TensorDataset(predictX, predicty)
        predict_dl = DataLoader(predict_db, batch_size=16, shuffle=False)
        predictX = predict_dl.dataset.tensors[0].numpy()
        return predictX

if __name__ == "__main__":
    model = ANFISDataLoader()
    print('shape: ', model.loadTrainData(train_path="RFANFIS_TRAIN_DATA.xlsx")[0].shape)
