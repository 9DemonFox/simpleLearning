import numpy as np

from data.hlm.dataloader import HLMDataLoader
from model import Model

# TODO: Finish HLM model by EM algorithm.
# ##---------------- test EM algorithm with GMM ------------------------## #


def generateData(k, mu, sigma, dataNum):
    """
    产生混合高斯模型的数据
    :param k: 比例系数
    :param mu: 均值
    :param sigma: 标准差
    :param dataNum:数据个数
    :return: 生成的数据
    """
    # 初始化数据
    dataArray = np.zeros(dataNum, dtype=np.float32)
    # 逐个依据概率产生数据
    # 高斯分布个数
    n = len(k)
    for i in range(dataNum):
        rand = np.random.random()  # [0, 1]
        Sum = 0
        index = 0
        while index < n:
            Sum += k[index]
            if rand < Sum:
                dataArray[i] = np.random.normal(mu[index], sigma[index])
                break
            else:
                index += 1
    return dataArray


def normPdf(x, mu, sigma):
    """
    计算均值为mu，标准差为sigma的正态分布函数的密度函数值
    :param x: x值
    :param mu: 均值
    :param sigma: 标准差
    :return: x处的密度函数值
    """
    return (1. / np.sqrt(2 * np.pi)) * (np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)))


def em(dataArray, k, mu, sigma, step=10):
    """
    em算法估计高斯混合模型
    :param dataNum: 已知数据个数
    :param k: 每个高斯分布的估计系数
    :param mu: 每个高斯分布的估计均值
    :param sigma: 每个高斯分布的估计标准差
    :param step:迭代次数
    :return: em 估计迭代结束估计的参数值[k,mu,sigma]
    """
    # 高斯分布个数
    n = len(k)
    # 数据个数
    dataNum = dataArray.size
    # 初始化gama数组
    gamaArray = np.zeros((n, dataNum))
    for s in range(step):
        # E step, 求参数的期望值
        # TODO: 求 gamma, sigma, T 的期望值
        for i in range(n):
            for j in range(dataNum):
                Sum = sum([k[t] * normPdf(dataArray[j], mu[t], sigma[t]) for t in range(n)])
                gamaArray[i][j] = k[i] * normPdf(dataArray[j], mu[i], sigma[i]) / float(Sum)

        # M step, 更新对应的参数值
        # TODO: 对应更新 gamma, sigma, T 的值
        # 更新 mu
        for i in range(n):
            mu[i] = np.sum(gamaArray[i] * dataArray) / np.sum(gamaArray[i])
        # 更新 sigma
        for i in range(n):
            sigma[i] = np.sqrt(np.sum(gamaArray[i] * (dataArray - mu[i]) ** 2) / np.sum(gamaArray[i]))
        # 更新系数k
        for i in range(n):
            k[i] = np.sum(gamaArray[i]) / dataNum

    return [k, mu, sigma]


# ##-----------------------------------------------------## #


class GBMModel(Model):
    def __init__(self, **kwargs):
        pass

    def fit(self, trainX, trainY):
        pass

    def predict(self, predictX):
        pass


if __name__ == "__main__":
    # 参数的准确值
    k = [0.3, 0.4, 0.3]
    mu = [2, 4, 3]
    sigma = [1, 1, 4]
    # 样本数
    dataNum = 5000
    # 产生数据
    dataArray = generateData(k, mu, sigma, dataNum)
    # 参数的初始值
    # 注意em算法对于参数的初始值是十分敏感的
    k0 = [0.3, 0.3, 0.4]
    mu0 = [1, 2, 2]
    sigma0 = [1, 1, 1]
    step = 6
    # 使用em算法估计参数
    k1, mu1, sigma1 = em(dataArray, k0, mu0, sigma0, step)
    # 输出参数的值
    print("参数实际值:")
    print("k:", k)
    print("mu:", mu)
    print("sigma:", sigma)
    print("参数估计值:")
    print("k1:", k1)
    print("mu1:", mu1)
    print("sigma1:", sigma1)
