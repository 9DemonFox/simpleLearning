import itertools
import numpy as np

import torch

from torch.utils.data import TensorDataset, DataLoader
from data.excelTools import ExcelTool

dtype = torch.float


def sinc(x1, x2, x3):
    '''
        Sinc is a simple two-input non-linear function
        used by Jang in section V of his paper (equation 30).
        修改为3个变量
    '''

    def s(z):
        return (1 if z == 0 else np.sin(z) / z)

    return s(x1) * s(x2) * s(x3)


def x_filter(x, y, sigma):
    x_np = x.numpy().T
    y_np = y.numpy().T
    # print(x_np.shape,y_np.shape)
    # for i in x_np:
    corration = np.corrcoef(x_np, y_np)
    tobedelete = []
    for i in range(x_np.shape[0]):
        if corration[i, x_np.shape[0]] <= sigma:
            tobedelete.append(i)
    x_np = np.delete(x_np, tobedelete, axis=0)
    return x_np.T


if __name__ == '__main__':
    num_cases = 100
    batch_size = 16
    pts = torch.linspace(-10, 10, int(np.sqrt(num_cases)))
    x = list(itertools.product(pts, repeat=3))
    x = torch.tensor(x, dtype=dtype)

    sigma = torch.min(x)
    y = torch.tensor([[sinc(*p)] for p in x], dtype=dtype)
    x = torch.from_numpy(x_filter(x, y, sigma))  # sigma需要视情况调整

    print(x.size(), y.size())

    data_np = np.concatenate((x.numpy(), y.numpy()), axis=1)
    trainX, trainY = data_np[0:900, :-1], data_np[0:900, -1]
    testX, testY = data_np[900:, :-1], data_np[900:, -1]
    predictX = data_np[900:901, :-1]

    ExcelTool.saveXY2Excel(trainX, trainY, "RFANFIS_TRAIN_DATA.xlsx")
    ExcelTool.saveXY2Excel(testX, testY, "RFANFIS_TEST_DATA.xlsx")
    ExcelTool.saveX2Excel(predictX, "RFANFIS_PREDICT_DATA.xlsx")

    # np.random.shuffle(data_np)
    #
    # trainX_np = data_np[0:900, :-1]
    # trainy_np = data_np[0:900, -1]
    # trainX_torch = torch.from_numpy(trainX_np)
    # trainy_torch = torch.from_numpy(trainy_np)
    # testX_np = data_np[900:,:-1]
    # testy_np = data_np[900:, -1]
    # testX_torch = torch.from_numpy(testX_np)
    # testy_torch = torch.from_numpy(testy_np)
