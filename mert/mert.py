import numpy as np
from sklearn import tree
from data.mert.dataLoder import MERTDataLoder
from model import Model


def update(trainX, trainY, epoch, n, N, m, D, q, u, σ2, clf, y0, z):
    """
    

    Parameters
    ----------   
    epoch:int
        循环轮数
    n : int
        观测对象种类数量
    N : int
        观测次数
    m : int
        每种观测对象的观测次数
    D : array
        随机效应变量的协方差矩阵
    q : int
        随机效应变量数
    u : array
        随机效应变量
    σ2 : float
        误差的方差
    clf : tree
        决策回归树
    y0 : array
    z : array
        随机效应参数

    Returns
    -------
    None.

    """
    for i in range(epoch):
        for j in range(n):
            y0[j * m:(j + 1) * m, 0:1] = trainY[j * m:(j + 1) * m, 0:1] - np.dot(z[j], u[j])
        clf.fit(trainX, y0)
        y1 = clf.predict(trainX).reshape(N, 1)
        for j in range(n):
            v = np.dot(np.dot(z[j], D), z[j].T) + σ2 * np.identity(m)
            u[j] = np.dot(np.dot(np.dot(D, z[j].T), np.linalg.inv(v)),
                          trainY[j * m:(j + 1) * m, 0:1] - y1[j * m:(j + 1) * m, 0:1])

        σ = 0.0
        d = np.zeros(q)
        for j in range(n):
            v = np.dot(np.dot(z[j], D), z[j].T) + σ2 * np.identity(m)
            e0 = trainY[j * m:(j + 1) * m, 0:1] - np.dot(z[j], u[j]) - y1[j * m:(j + 1) * m, 0:1]
            σ = σ + np.dot(e0.T, e0)  # + σ2*(m - σ2*np.trace(v))
            d = d + np.dot(u[j], u[j].T) + D - np.dot(np.dot(np.dot(np.dot(D, z[j].T), np.linalg.inv(v)), z[j]), D)
        σ2 = σ / N
        D = d / n


class MERTModel(Model):

    def __init__(self, **kwargs):
        """
        
        n:int,可选(默认 = 1)
        观测对象种类数量
        q：int
        随机效应变量数
        D:array
        随机效应变量的协方差矩阵
        u:array
        随机效应变量
        σ2:float
        误差的方差

        """
        if "n" not in kwargs.keys():
            self.n = 1
        else:
            self.n = kwargs["n"]
        self.q = 2
        self.D = np.identity(self.q)
        self.u = np.zeros((self.n, self.q, 1))
        self.σ2 = 1.0
        self.clf = tree.DecisionTreeRegressor(criterion='mse', max_depth=(10))

    def fit(self, **kwargs):

        """
        
        epoch:int,可选(默认=200)
        循环轮数
        N:int
        观测次数
        m:int
        每种观测对象的观测次数
        z:array
        随机效应参数

        """
        trainX = kwargs["trainX"]
        trainY = kwargs["trainY"]
        if "epoch" not in kwargs.keys():
            epoch = 200
        else:
            epoch = kwargs["epoch"]

        N = trainY.shape[0]
        m = int(N / self.n)
        z = np.ones((self.n, m, self.q))
        for i in range(self.n):
            z[i:i + 1, :, 1:2] = z[i:i + 1, :, 1:2] - 1 + trainX[m * i:m * (i + 1), 0:1]
        y0 = np.empty((N, 1))
        update(trainX, trainY, epoch, self.n, N, m, self.D, self.q, self.u, self.σ2, self.clf, y0, z)
        for j in range(self.n):
            y0[j * m:(j + 1) * m, 0:1] = trainY[j * m:(j + 1) * m, 0:1] - np.dot(z[j], self.u[j])

        self.clf2 = tree.DecisionTreeRegressor(criterion='mse', random_state=0, ccp_alpha=0.1, max_depth=(10))
        self.clf2.fit(trainX, y0)
        # tree.plot_tree(self.clf2,filled=True)
        x0 = self.clf2.predict(trainX).reshape(N, 1)
        for j in range(self.n):
            x0[j * m:(j + 1) * m, 0:1] = x0[j * m:(j + 1) * m, 0:1] + np.dot(z[j], self.u[j])

    def predict(self, **kwargs):
        predictX = kwargs["predictX"]
        predictY = kwargs["predictY"]
        N = predictY.shape[0]
        m = int(N / self.n)
        z = np.ones((self.n, m, self.q))
        for i in range(self.n):
            z[i:i + 1, :, 1:2] = z[i:i + 1, :, 1:2] - 1 + predictX[m * i:m * (i + 1), 0:1]
        x0 = self.clf2.predict(predictX).reshape(N, 1)
        for j in range(self.n):
            x0[j * m:(j + 1) * m, 0:1] = x0[j * m:(j + 1) * m, 0:1] + np.dot(z[j], self.u[j])
        return x0


if __name__ == '__main__':
    n = 100
    epoch = 50
    model = MERTModel(n=n)
    dataloader = MERTDataLoder(datapath1="../data/mert/data_train.csv", datapath2="../data/mert/data_test.csv")
    trainX, trainY = dataloader.loadTrainData()

    predictX, predictY = dataloader.loadTestData()
    model.fit(trainX=trainX, trainY=trainY, epoch=epoch)
    model.predict(predictX=predictX, predictY=predictY)
