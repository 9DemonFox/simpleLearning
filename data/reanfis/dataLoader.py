import sys
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from collections import OrderedDict

dtype = torch.float


def sinc( x1, x2, x3):
    '''
        Sinc is a simple two-input non-linear function
        used by Jang in section V of his paper (equation 30).
        修改为3个变量
    '''

    def s(z):
        return (1 if z == 0 else np.sin(z) / z)

    return s(x1) * s(x2) * s(x3)


def x_filter( x, y, sigma):
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


class anfisDataLoader(DataLoader):
    def __init__(self, num_cases=100, batch_size=16):
        pts = torch.linspace(-10, 10, int(np.sqrt(num_cases)))
        x = list(itertools.product(pts, repeat=3))
        x = torch.tensor(x, dtype=dtype)

        print(x.numpy())

        sigma = torch.min(x)
        y = torch.tensor([[sinc(*p)] for p in x], dtype=dtype)
        x = torch.from_numpy(x_filter(x, y, sigma))  # sigma需要视情况调整

        print(x.size(), y.size())
        print(type(x),type(y))

        #将x,y转成array，构造train/test再转回
        data_np = np.concatenate((x.numpy(),y.numpy()),axis=1)
        np.random.shuffle(data_np)

        trainX_np = data_np[0:900, :-1]
        trainy_np = data_np[0:900, -1]
        trainX_torch = torch.from_numpy(trainX_np)
        trainy_torch = torch.from_numpy(trainy_np)
        testX_np = data_np[900:,:-1]
        testy_np = data_np[900:, -1]
        testX_torch = torch.from_numpy(testX_np)
        testy_torch = torch.from_numpy(testy_np)

        train_db = TensorDataset(trainX_torch, trainy_torch)
        test_db = TensorDataset(testX_torch, testy_torch)


        #data = torch.cat((x,y), 1)
        #td = TensorDataset(x, y)
        #train_db, test_db = torch.utils.data.random_split(td, [900, 100])

        #self.dl = DataLoader(td, batch_size=batch_size, shuffle=True)
        self.train_dl = DataLoader(train_db, batch_size=batch_size, shuffle=True)
        self.test_dl = DataLoader(test_db, batch_size=batch_size, shuffle=True)

        #dataiter = iter(dl)
        #print(dataiter.next())


    def loadTrainData(self):
        return self.train_dl

    def loadTestData(self):
        return self.test_dl

if __name__ == "__main__":
    model = anfisDataLoader(100,16)
    print(model.loadTrainData())