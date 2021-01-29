import numpy as np
from sklearn import tree
from data.mert.dataLoder import MERTDataLoader
from model import Model

from sklearn.metrics import mean_squared_error

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
    
    return D,σ2


class MERTModel(Model):

    def __init__(self, **kwargs):
        """
        
        n:int,可选(默认 = 100)
        观测对象种类数量
        q：int
        随机效应变量数
        D:array
        随机效应变量的协方差矩阵
        u:array
        随机效应变量
        σ2:float
        误差的方差 
        epoch:int,可选(默认=50)
        循环轮数
        k:int,可选(默认=1)
        表示第k个变量作为随机效应变量

        """
        if "n" not in kwargs.keys():
            self.n = 100
        else:
            self.n = kwargs["n"]
        if "epoch" not in kwargs.keys():
            self.epoch = 50
        else:
            self.epoch = kwargs["epoch"]

        if "k" not in kwargs.keys():
            self.k = 1
        else:
            self.k = kwargs["k"]
        self.q = 2
        self.D = np.identity(self.q)
        self.u = np.zeros((self.n, self.q, 1))
        self.σ2 = 1.0
        self.clf = tree.DecisionTreeRegressor(criterion='mse', max_depth=(10))

    def fitForUI(self, **kwargs):
        """ 返回结果到前端
        :return:
        """
        # 返回结果为字典形式
        D,σ2 = self.fit(**kwargs)
        returnDic = {
            "随机效应协方差矩阵": str(D),
            "误差方差": str(σ2)
        }
        return returnDic
    
    def testForUI(self, **kwargs):
        """
        :param kwargs:
        :return: 字典形式结果
        """
        returnDic = {
            "mse": None
        }
        predictResult = self.test(**kwargs)
        returnDic["mse"] = str(predictResult)
        return returnDic
    
    def predictForUI(self, **kwargs):
        """
        :param kwargs:
        :return: 字典形式结果
        """
        returnDic = {
            "预测结果": None
        }
        predictResult = self.predict(**kwargs)
        returnDic["预测结果"] = str(predictResult)
        return returnDic
    
    def fit(self, **kwargs):

        """
        
       
        N:int
        观测次数
        m:int
        每种观测对象的观测次数
        z:array
        随机效应参数

        """
        self.trainX = kwargs["trainX"]
        self.trainY = kwargs["trainY"]
        N = self.trainY.shape[0]
        self.trainY = self.trainY.reshape(N,1)
        m = int(N / self.n)
        z = np.ones((self.n, m, self.q))
        for i in range(self.n):
            z[i:i + 1, :, 1:2] = z[i:i + 1, :, 1:2] - 1 + self.trainX[m * i:m * (i + 1), self.k-1:self.k]
        y0 = np.empty((N, 1))
        D,σ2 = update(self.trainX, self.trainY, self.epoch, self.n, N, m, self.D, self.q, self.u, self.σ2, self.clf, y0, z)
        for j in range(self.n):
            y0[j * m:(j + 1) * m, 0:1] = self.trainY[j * m:(j + 1) * m, 0:1] - np.dot(z[j], self.u[j])

        self.clf2 = tree.DecisionTreeRegressor(criterion='mse', random_state=0, ccp_alpha=0.1, max_depth=(10))
        self.clf2.fit(self.trainX, y0)
        # tree.plot_tree(self.clf2,filled=True)
        x0 = self.clf2.predict(self.trainX).reshape(N, 1)
        for j in range(self.n):
            x0[j * m:(j + 1) * m, 0:1] = x0[j * m:(j + 1) * m, 0:1] + np.dot(z[j], self.u[j])
        
        return D,σ2

    def test(self, **kwargs):
        self.testX = kwargs["testX"]
        self.testY = kwargs["testY"]
        N = self.testX.shape[0]
        m = int(N / self.n)
        z = np.ones((self.n, m, self.q))
        for i in range(self.n):
            z[i:i + 1, :, 1:2] = z[i:i + 1, :, 1:2] - 1 + self.testX[m * i:m * (i + 1), self.k-1:self.k]
        x0 = self.clf2.predict(self.testX).reshape(N, 1)
        for j in range(self.n):
            x0[j * m:(j + 1) * m, 0:1] = x0[j * m:(j + 1) * m, 0:1] + np.dot(z[j], self.u[j])
        return mean_squared_error(x0, self.testY)
    
    def predict(self, **kwargs):
        self.predictX = kwargs["predictX"]
        N = self.predictX.shape[0]
        m = int(N / self.n)
        z = np.ones((self.n, m, self.q))
        for i in range(self.n):
            z[i:i + 1, :, 1:2] = z[i:i + 1, :, 1:2] - 1 + self.predictX[m * i:m * (i + 1), self.k-1:self.k]
        x0 = self.clf2.predict(self.predictX).reshape(N, 1)
        for j in range(self.n):
            x0[j * m:(j + 1) * m, 0:1] = x0[j * m:(j + 1) * m, 0:1] + np.dot(z[j], self.u[j])
        return x0


if __name__ == '__main__':
    n = 100
    epoch = 50
    k = 1
    datapath1="../data/mert/data_train.xlsx"
    datapath2="../data/mert/data_test.xlsx"
    datapath3="../data/mert/data_predict.xlsx"
    dataloader = MERTDataLoader()
    trainX, trainY = dataloader.loadTrainData(train_path=datapath1)
    textX, textY = dataloader.loadTestData(test_path=datapath2)
    predictX = dataloader.loadPredictData(predict_path=datapath3)
    model = MERTModel(n=n, epoch=epoch, k=k)
    print(model.fitForUI(trainX=trainX, trainY=trainY))
    print(model.testForUI(testX=textX, testY=textY))
    print(model.predictForUI(predictX=textX))
