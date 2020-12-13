import numpy as np
from sklearn import tree
from data.rebet.dataLoder import REBETDataLoder
from model import Model
from scipy.stats import gamma
import math
from scipy import stats


def update(trainX, trainY, epoch, n, N, m, D, q, u, σ2, clf, y0, z, M):
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
    D:  array
        迪利克雷参数中的协方差矩阵
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
    M:  float
        迪利克雷分布参数

    Returns
    -------
    None.

    """
    u[0] = np.random.multivariate_normal(np.zeros(q), D, size=(1)).reshape(2, 1)

    for i in range(n):
        if i != 0:
            j = i + 1
            a = np.random.binomial(1, M / (M + j - 1), 1)
            if a == 1:
                u[i] = np.random.multivariate_normal(np.zeros(q), D, size=(1)).reshape(2, 1)

            else:
                b = np.random.randint(1, j)
                u[i] = u[b - 1]

    for i in range(epoch):
        for j in range(n):
            y0[j * m:(j + 1) * m, 0:1] = trainY[j * m:(j + 1) * m, 0:1] - np.dot(z[j], u[j])
        clf.fit(trainX, y0)
        y1 = clf.predict(trainX).reshape(N, 1)

        a = 0
        for j in range(n):
            a = a + np.dot((trainY[j * m:(j + 1) * m, 0:1] - y1[j * m:(j + 1) * m, 0:1] - np.dot(z[j], u[j])).T,
                           (trainY[j * m:(j + 1) * m, 0:1] - y1[j * m:(j + 1) * m, 0:1] - np.dot(z[j], u[j])))

        a = a / 2
        b = N / 2 + 1
        σ2 = 1 / gamma.rvs(b, scale=1 / a)
        r = np.empty((n, q, 1))
        k = 0

        l = np.zeros(n)
        c = 0
        for t in range(n):
            f = stats.multivariate_normal.pdf((trainY[t * m:(t + 1) * m, 0:1]).reshape(m),
                                              mean=(y1[t * m:(t + 1) * m, 0:1] + np.dot(z[t], u[t])).reshape(m),
                                              cov=σ2 * np.identity(m))
            c = c + stats.multivariate_normal.pdf((trainY[t * m:(t + 1) * m, 0:1]).reshape(m),
                                                  mean=(y1[t * m:(t + 1) * m, 0:1] + np.dot(z[t], u[t])).reshape(m),
                                                  cov=σ2 * np.identity(m))
            l[t] = f

        for j in range(n):
            a = np.identity(q)
            d = np.identity(q)
            Q = np.linalg.inv(np.dot(z[j].T, z[j]) / σ2 + np.divide(a, D, out=np.zeros_like(a), where=D != 0))

            U = np.dot(np.dot(z[j], Q), z[j].T) / σ2 - np.identity(m)
            A = np.linalg.inv(D * np.identity(q))
            B = Q * np.identity(q)
            I = M * math.pow(σ2 * 2 * math.pi, m / (-2)) * np.power(A, 1 / 2) * np.power(B, 1 / 2) * math.exp((np.dot(
                np.dot((trainY[j * m:(j + 1) * m, 0:1] - y1[j * m:(j + 1) * m, 0:1]).T, U),
                trainY[j * m:(j + 1) * m, 0:1] - y1[j * m:(j + 1) * m, 0:1])) / (2 * σ2))

            w = math.pow(np.linalg.det(I), 1 / q)
            f = c - l[j]
            p = w / (w + f)
            if np.isnan(p) or p < 0:
                p = 0
            if np.isinf(p) or p > 1:
                p = 1
            a = np.random.binomial(1, p, 1)
            if a == 1:
                d = np.identity(q)
                mean = np.dot(np.dot(
                    np.linalg.inv(np.dot(z[j].T, z[j]) + σ2 * np.divide(d, D, out=np.zeros_like(d), where=D != 0)),
                    z[j].T), trainY[j * m:(j + 1) * m, 0:1] - y1[j * m:(j + 1) * m, 0:1])
                d = np.identity(q)
                cov = σ2 * np.linalg.inv(
                    np.dot(z[j].T, z[j]) + σ2 * np.divide(d, D, out=np.zeros_like(d), where=D != 0))
                u[j] = np.random.multivariate_normal(mean.reshape(q), cov, size=(1)).reshape(2, 1)
                r[k] = u[j]
                k = k + 1
            else:
                s = np.delete(l, j)
                b = np.random.choice(np.arange(n - 1), size=1, replace=True,
                                     p=(s) / (s.sum()))

                if b < j:
                    u[j] = u[b]
                else:
                    u[j] = u[b + 1]
                a = 0
                for t in range(k):
                    if (r[t] == u[j]).all():
                        a = 1
                        break
                if a == 0:
                    r[k] = u[j]
                    k = k + 1
        r = r[0:k, :, :].reshape(k, q)

        a = np.dot(r.T, r) / 2
        b = k / 2 + 1
        for j in range(q):
            for t in range(q):
                if (j < t):
                    D[j][t] = 1 / gamma.rvs(b, scale=np.abs(1 / a[j][t]))
                    D[t][j] = D[j][t]
                if (j == t):
                    D[j][t] = 1 / gamma.rvs(b, scale=np.abs(1 / a[j][t]))


class REBETModel(Model):

    def __init__(self, **kwargs):
        """
        
        n:int,可选(默认 = 1)
        观测对象种类数量
        q：int
        随机效应变量数
        D:array
        迪利克雷参数中的协方差矩阵
        u:array
        随机效应变量
        σ2:float
        误差的方差
        M:float
        迪利克雷分布参数

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
        self.M = 10

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
        update(trainX, trainY, epoch, self.n, N, m, self.D, self.q, self.u, self.σ2, self.clf, y0, z, self.M)
        for j in range(self.n):
            y0[j * m:(j + 1) * m, 0:1] = trainY[j * m:(j + 1) * m, 0:1] - np.dot(z[j], self.u[j])

        self.clf2 = tree.DecisionTreeRegressor(criterion='mse', random_state=0, ccp_alpha=0.1, max_depth=(10))
        self.clf2.fit(trainX, y0)
        #tree.plot_tree(self.clf2, filled=True)
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
    model = REBETModel(n=n)
    dataloader = REBETDataLoder(datapath1="../data/mert/data_train.csv", datapath2="../data/mert/data_test.csv")
    trainX, trainY = dataloader.loadTrainData()

    predictX, predictY = dataloader.loadTestData()
    model.fit(trainX=trainX, trainY=trainY, epoch=epoch)
    model.predict(predictX=predictX, predictY=predictY)
