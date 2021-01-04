# 用于构造论文中数据集

import numpy
import pandas
from numpy.random import normal as Normal
from numpy.random import uniform as Uniform

from data.excelTools import ExcelTool


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


def saveXY2Excel(x, y, path="SALP_TRAIN_DATA.xlsx"):
    xy = numpy.insert(x, 0, values=y, axis=1)
    df = pandas.DataFrame.from_records(xy)
    columns = ["y"]
    columns_x = ["x" + str(i) for i in range(trainX.shape[1])]
    columns.extend(columns_x)
    df.columns = columns
    df.to_excel(path)


if __name__ == '__main__' and True:
    """ 
    生成测试集和训练集
    """
    x, y = get_sample_data()
    trainX, trainY = x[0:80, :], y[0:80].reshape(1, -1)
    testX, testY = x[80:, :], y[80:].reshape(1, -1)
    predictX, predictY = x[80:81, :], y[80:81].reshape(1, -1)
    ExcelTool.saveXY2Excel(trainX, trainY, "SALP_TRAIN_DATA.xlsx")
    ExcelTool.saveXY2Excel(testX, testY, "SALP_TEST_DATA.xlsx")
    ExcelTool.saveX2Excel(predictX, "SALP_PREDICT_DATA.xlsx")

if __name__ == '__main__' and False:
    x, y = get_sample_data()


    def standardrized(data):
        mean = data.mean()  # 计算平均数
        deviation = data.std()  # 计算标准差
        # 标准化数据的公式: (数据值 - 平均数) / 标准差
        std_data = (data - mean) / deviation
        return std_data


    std_y = standardrized(y)
    std_x = numpy.apply_along_axis(standardrized, axis=1, arr=x)
    print((std_y.mean(), std_y.std()), (std_x[0].mean(), std_x[0].std()))
    from sklearn.linear_model import LassoLars

    clf = LassoLars(alpha=0.1, fit_intercept=False)
    clf.fit(std_x, std_y)
    print(clf.coef_)
