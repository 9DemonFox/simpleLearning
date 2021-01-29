import os

import numpy
import pandas
from numpy.random import normal as Normal
from numpy.random import uniform as Uniform

from data.dataLoader import DataLoader


def get_sample_data():
    n = 100  # 样本容量
    k = 10  # 目标变量数量
    d = 100  # 特征数量
    delta = 5
    # L1 - Lk
    sigma = 1 / 3  # 为调整系数， 用于调整信号和噪声的幅度比值
    E = 0.1
    tau = numpy.full((n, k), 0.3)  # y的离群程度 [样本容量,目标变量数量k]
    epsilon_1 = Normal(0, 1)
    epsilon_2 = (1 - E) * Normal(0, 1) + E * Normal(0, 1) / Uniform(0, 1)
    epsilon_3 = (1 - E) * Normal(0, 1) + E * Normal(20, 1)
    epsilon_4 = (1 - E) * Normal(0, 1) + E * Normal(50, 1) / Uniform(50, 1)
    epsilon_5 = (1 - E) * Normal(0, 1) + E * Normal(50, 1)
    epsilon = epsilon_1
    L = Normal(0, 1, (n, k))  # k个L数
    y = L.sum(axis=1) + sigma * epsilon
    delta = 5
    tau = numpy.full((n, k), 0.3)
    e = Uniform(0, 1, (n, d))
    x = numpy.zeros((n, d))
    x[:, 0:k] = L + tau * e[:, 0:k]
    x[:, k:3 * k:2] = L + delta * e[:, k:3 * k:2]
    x[:, k + 1:3 * k + 1:2] = L + delta * e[:, k + 1:3 * k + 1:2]
    x[:, 3 * k + 1:d] = e[:, 3 * k + 1:d]
    return x, y


def generateData():
    n = 100
    d = 100
    salpData = numpy.zeros([5, n, d + 1])
    for i in range(5):  # 对于每个生成函数
        for j in range(n):  # 对于每个Xy
            x, y = get_sample_data()
            xy = numpy.insert(x, 0, values=y, axis=1)  # 合并x,y
            salpData[i] = xy
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


class SALPDataLoader(DataLoader):
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
        assert "train_path" in kwargs.keys()
        trainX, trainY = self.__loadExcelData(kwargs.get("train_path"))
        return trainX, trainY

    def loadTestData(self, **kwargs):
        assert "test_path" in kwargs.keys()
        testX, testY = self.__loadExcelData(kwargs.get("test_path"))
        return testX, testY

    def loadPredictData(self, **kwargs):
        assert "predict_path" in kwargs.keys()
        df = pandas.read_excel(kwargs.get("predict_path"), index_col=0)
        predictX = df.values[:, :]
        return predictX

if __name__ == "__main__":
    from sklearn.svm import SVR
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LassoLars
    import numpy as np
    from sklearn.linear_model import LassoLars

    # 自动选择合适的参数
    svr = GridSearchCV(SVR(),
                       param_grid={"kernel": ("linear", 'rbf'), "C": np.logspace(-3, 3, 7),
                                   "gamma": np.logspace(-3, 3, 7)})
    # lasso = GridSearchCV(LinearRegression(),
    #                      param_grid={
    #                          "fit_intercept": [True, False]
    #                      }
    # param_grid={
    #     "alpha": [0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.03, 0.06, 0.09, 0.12, 0.15,
    #               0.2, 0.3,
    #               0.4, 0.5]}
    # )

    dataLoader = SALPDataLoader("SALP_DATA.npy")
    trainX, trainY = dataLoader.loadTrainData(train_path="SALP_TRAIN_DATA.xlsx")
    testX, testY = dataLoader.loadTestData(test_path="SALP_TEST_DATA.xlsx")


    # def normalXY(x, y):
    #     """标准化 均值为0 平方和为1
    #     :param x:
    #     :param y:
    #     :return:
    #     """
    #     normal_l2 = lambda data: normalize(data.reshape(1, -1), norm="l2").squeeze()
    #     y = normal_l2(y)
    #     x = numpy.apply_along_axis(normal_l2, axis=1, arr=x)
    #     return x, y

    def normalXY_2(x, y):
        def standardrized(data):
            mean = data.mean()  # 计算平均数
            deviation = data.std()  # 计算标准差
            # 标准化数据的公式: (数据值 - 平均数) / 标准差
            std_data = (data - mean) / deviation
            return std_data

        std_y = standardrized(y)
        std_x = numpy.apply_along_axis(standardrized, axis=1, arr=x)
        return std_x, std_y


    stdX, stdY = normalXY_2(trainX, trainY)

    # svr.fit(trainX, trainY)
    # 标准数据集
    lasso = LassoLars(alpha=0.05)
    lasso.fit(trainX, trainY)
    predictY = lasso.predict(testX[0].reshape(1, -1))
    print(lasso.coef_)
    # 标准化数据
    lasso = LassoLars(alpha=0.08, fit_intercept=False)
    lasso.fit(stdX, stdY)
    predictY = lasso.predict(testX[0].reshape(1, -1))
    print(lasso.coef_)
    # print(lasso.best_estimator_)
    # err = mean_squared_error(predictY, testY)
    # print(err)

    # svr.fit(trainX, trainY)
    # predictY = svr.predict(testX)
    # # print(svr.best_estimator_.coef_)
    # # print(lasso.best_estimator_)
    # err = mean_squared_error(predictY, testY)
    # print(err)
