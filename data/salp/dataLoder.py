import os

import numpy
from numpy.random import normal as Normal
from numpy.random import uniform as Uniform
from sklearn.preprocessing import normalize

DATAPATH = "SALP_DATA.npy"
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
    # 数据标准化
    y = normal_l2(y)
    x = numpy.apply_along_axis(normal_l2, axis=1, arr=x)
    return x, y


# 按照
def normal_l2(data):
    # 标准化 均值为0 平方和为1
    return normalize(data.reshape(1, -1), norm="l2").squeeze()


def bayesian_bootstrap(X,
                       statistic,
                       n_replications,
                       resample_size,
                       low_mem=False):
    """Simulate the posterior distribution of the given statistic.

    Parameter X: The observed data (array like)

    Parameter statistic: A function of the data to use in simulation (Function mapping array-like to number)

    Parameter n_replications: The number of bootstrap replications to perform (positive integer)

    Parameter resample_size: The size of the dataset in each replication

    Parameter low_mem(bool): Generate the weights for each iteration lazily instead of in a single batch. Will use
    less memory, but will run slower as a result.

    Returns: Samples from the posterior
    """
    if isinstance(X, list):
        X = numpy.array(X)
    samples = []
    samples_index = []
    if low_mem:
        weights = (numpy.random.dirichlet([1] * len(X))
                   for _ in range(n_replications))
    else:
        weights = numpy.random.dirichlet([1] * len(X), n_replications)
    for w in weights:
        sample_index = numpy.random.choice(range(len(X)), p=w, size=resample_size)
        samples_index.append(sample_index)
        resample_X = X[sample_index]
        s = statistic(resample_X)
        samples.append(s)
    return samples, samples_index


def split_xy(xy):
    x = xy[:, 1:]
    y = xy[:, 0]
    return x, y


def merge_xy(x, y):
    return numpy.insert(x, 0, values=y, axis=1)


def getBayesianBootstrapReconstructData(x, y, n_replications):
    """
    使用bayesian_tootstrap重构数据集
    :param x:
    :param y:
    :param n_replications:
    :return:
    """
    std_xy = numpy.insert(x, 0, values=y, axis=1)  # 合并x,y
    bayes_xys, bayes_indexs = bayesian_bootstrap(std_xy,
                                                 lambda x: x,
                                                 n_replications=n_replications,
                                                 resample_size=len(std_xy))
    xs = []
    ys = []
    for i in range(n_replications):
        x, y = split_xy(bayes_xys[i])
        xs.append(x)
        ys.append(y)
    return (xs, ys, bayes_indexs)


def generateData():
    salpData = numpy.zeros([len(Epsilons), dataset_num, n, d + 1])
    salpData.shape
    for i, epsilon in enumerate(Epsilons):  # 对于每个生成函数
        for j in range(n):  # 样本容量
            x, y = get_sample_data(epsilon)
            xy = numpy.insert(x, 0, values=y, axis=1)  # 合并x,y
            salpData[i][j] = xy
    return salpData


def loadSALPData():
    # 数据已经标准化处理
    if os.path.exists(DATAPATH):
        salp = numpy.load(DATAPATH)
    else:
        salp = generateData()
        numpy.save(DATAPATH, salp)
    return salp


from data.dataLoder import DataLoder


class SalpDataLoder(DataLoder):
    def __init__(self):
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
        dataSet = loadSALPData()
        trainX = dataSet[degreeOfOutliers, 0:80, :, :-1]
        trainY = dataSet[degreeOfOutliers, 0:80, :, -1]
        return trainX[0], trainY[0]

    def loadTestData(self, **kwargs):
        if "degreeOfOutliers" in kwargs.keys():
            degreeOfOutliers = int(kwargs["degreeOfOutliers"])
            assert degreeOfOutliers >= 1 and degreeOfOutliers <= 5
        else:
            degreeOfOutliers = 2
        degreeOfOutliers = degreeOfOutliers - 1  # index
        dataSet = loadSALPData()
        testX = dataSet[degreeOfOutliers, 80:, :, :-1]
        testY = dataSet[degreeOfOutliers, 80:, :, -1]
        return testX[0], testY[0]


# svr.fit(x, y)
if __name__ == "__main__":
    from sklearn.svm import SVR
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import GridSearchCV
    import numpy as np

    rng = np.random
    # svr = joblib.load('svr.pkl')        # 读取模型

    x = rng.uniform(1, 100, (100, 1))
    y = 5 * x + np.sin(x) * 5000 + 2 + np.square(x) + rng.rand(100, 1) * 5000

    # 自动选择合适的参数
    svr = GridSearchCV(SVR(),
                       param_grid={"kernel": ("linear", 'rbf'), "C": np.logspace(-3, 3, 7),
                                   "gamma": np.logspace(-3, 3, 7)})
    dataloader = SalpDataLoder()
    trainX, trainY = dataloader.loadTrainData()
    testX, testY = dataloader.loadTestData()
    svr.fit(trainX[1], trainY[1])
    predictY = svr.predict(testX[1])
    err = mean_squared_error(predictY, testY[1])
    print()
    pass
