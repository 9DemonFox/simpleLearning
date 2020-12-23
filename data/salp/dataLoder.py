import os

import numpy
from numpy.random import normal as Normal
from numpy.random import uniform as Uniform

from data.dataLoader import DataLoader

GROUPINDEX = "INDEX.npy"

dataset_num = 100  # 数据集数量
n = 100  # 样本容量
k = 10  # 目标变量
d = 100  # 变量数量
E = 0.1
delta = 5
tau = numpy.full((n, k), 0.3)
# 不同程度的 epsilon
epsilon_1 = lambda: Normal(0, 1)
epsilon_2 = lambda: (1 - E) * Normal(0, 1) + E * Normal(0, 1) / Uniform(0, 1)
epsilon_3 = lambda: (1 - E) * Normal(0, 1) + E * Normal(20, 1)
epsilon_4 = lambda: (1 - E) * Normal(0, 1) + E * Normal(50, 1) / Uniform(50, 1)
epsilon_5 = lambda: (1 - E) * Normal(0, 1) + E * Normal(50, 1)
Epsilons = [epsilon_1, epsilon_2, epsilon_3, epsilon_4, epsilon_5]


# 获取 epsilon 表达了y的离群度
def get_sample_data(epsilon):
    # L1 - Lk
    sigma = 1 / 3  # 为调整系数， 用于调整信号和噪声的幅度比值
    L = Normal(0, 1, (n, k))  # k个L数
    y = L.sum(axis=1) + sigma * epsilon()
    e = Uniform(0, 1, (n, d))
    x = numpy.zeros((n, d))
    x[:, 0:k] = L + tau * e[:, 0:k]
    x[:, k:3 * k:2] = L + delta * e[:, k:3 * k:2]
    x[:, k + 1:3 * k + 1:2] = L + delta * e[:, k + 1:3 * k + 1:2]
    x[:, 3 * k + 1:d] = e[:, 3 * k + 1:d]
    return x, y


def generateData():
    salpData = numpy.zeros([len(Epsilons), dataset_num, n, d + 1])
    for i, epsilon in enumerate(Epsilons):  # 对于每个生成函数
        for j in range(n):  # 样本容量
            x, y = get_sample_data(epsilon)
            xy = numpy.insert(x, 0, values=y, axis=1)  # 合并x,y
            salpData[i][j] = xy
    return salpData


def loadSALPData(data_path="SALP_DATA.npy"):
    """ 存在数据就不处理，不存在就生成数据
    :return:
    """
    if os.path.exists(data_path):
        salp = numpy.load(data_path)
    else:
        salp = generateData()
        numpy.save(data_path, salp)
    return salp


class SalpDataLoader(DataLoader):
    def __init__(self, data_path):
        self.data_path = data_path
        pass

    def loadTrainData(self, **kwargs):
        """
        :param kwargs: degreeOfOutliers 离群程度 1-5  referPaper/基于数据挖掘的材料自然环境腐蚀预测研究.pdf
        :return:
        """
        degreeOfOutliers = 1
        if "degreeOfOutliers" in kwargs.keys():
            degreeOfOutliers = int(kwargs["degreeOfOutliers"])
            assert degreeOfOutliers >= 1 and degreeOfOutliers <= 5
        else:
            degreeOfOutliers = 2
        degreeOfOutliers = degreeOfOutliers - 1  # index
        dataSet = loadSALPData(self.data_path)
        trainX = dataSet[degreeOfOutliers, 0:80, :, :-1]
        trainY = dataSet[degreeOfOutliers, 0:80, :, -1]
        return trainX[0][:80], trainY[0][:80]

    def loadTestData(self, **kwargs):
        if "degreeOfOutliers" in kwargs.keys():
            degreeOfOutliers = int(kwargs["degreeOfOutliers"])
            assert degreeOfOutliers >= 1 and degreeOfOutliers <= 5
        else:
            degreeOfOutliers = 2  # 默认离群程度
        degreeOfOutliers = degreeOfOutliers - 1  # index
        dataSet = loadSALPData(self.data_path)
        testX = dataSet[degreeOfOutliers, 80:, :, :-1]
        testY = dataSet[degreeOfOutliers, 80:, :, -1]
        return testX[0][80:], testY[0][80:]


if __name__ == "__main__":
    from sklearn.svm import SVR
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import GridSearchCV
    import numpy as np

    # 自动选择合适的参数
    svr = GridSearchCV(SVR(),
                       param_grid={"kernel": ("linear", 'rbf'), "C": np.logspace(-3, 3, 7),
                                   "gamma": np.logspace(-3, 3, 7)})
    dataloader = SalpDataLoader("SALP_DATA.npy")
    trainX, trainY = dataloader.loadTrainData()
    testX, testY = dataloader.loadTestData()
    svr.fit(trainX, trainY)
    predictY = svr.predict(testX)
    err = mean_squared_error(predictY, testY)
