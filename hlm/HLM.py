import numpy as np
from sklearn.metrics import mean_squared_error

from data.hlm.dataloader import HLMDataLoader
from model import Model

# global params:
SEED = 134
DECIMALS = 6

# beta_sample number:
M = 80


def my_debug(var, name, flag):
    print("{}.shape is ".format(name), var.shape)
    if flag is True:
        print(var)


def mat_inv(X):
    """
    matrix inversion.
    """
    X = np.around(X, decimals=DECIMALS)
    return np.linalg.inv(X)


def mat_around(X):
    """
    around the matrix values.
    """
    return np.around(X, decimals=DECIMALS)


def init_params(p, q):
    """ 随机初始的 gamma, sigma_squ, T.
    :param p: shape of T
    :param q: shape of gamma
    """
    np.random.seed(SEED)
    gamma = np.mat(np.random.uniform(0, 1, (q, 1)))
    np.random.seed(SEED)
    sigma_squ = np.random.uniform(0, 1)  # scalar
    np.random.seed(SEED)

    # generate positive definite mat
    A = np.random.rand(p, p)
    B = np.dot(A, A.transpose())
    C = B + B.T
    T = np.mat(C) / 10
    T[0, 1] = 0
    T[1, 0] = 0

    # T = np.mat(np.random.uniform(0, 1, (p, p)))
    return gamma, sigma_squ, T


def em_hlm(W, X, Y, iters):
    """ EM algorithm to estimate params, include gamma, sigma_squ, T.
    :param W: 第二层的固定效应参数矩阵。
    :param X: 第一层的自变量参数矩阵。
    :param Y: 第一层的因变量参数矩阵。
    :param iters: 最大迭代次数。

    :return: gamma, sigma_squ, T.
    """
    Y = np.mat(Y)
    N = Y.shape[0]

    X = np.mat(X)
    W = np.mat(W)

    p = X.shape[1]
    q = W.shape[1]
    # print(p, q)

    # my_debug(Y, 'Y', True), my_debug(X, 'X', True), my_debug(W, 'W', True)

    gamma, sigma_squ, T = init_params(p, q)
    # my_debug(gamma, 'gamma', True)
    # my_debug(T, 'T', True)
    mu = np.random.multivariate_normal(mean=np.zeros(p), cov=T, size=N)
    mu = np.mat(mu)
    mu = mat_around(mu)
    # my_debug(mu, 'mu', True)

    for i in range(iters):
        beta_mean = W * gamma + mat_inv(X.T * X + sigma_squ * mat_inv(T)) * X.T * (Y - X * (W * gamma))  # shape(2, 1)
        beta_mean = mat_around(beta_mean)

        beta_cov = sigma_squ * mat_inv(X.T * X + sigma_squ * mat_inv(T))  # shape(2, 2)
        beta_cov = mat_around(beta_cov)
        # my_debug(beta_cov, 'beta_cov', True), my_debug(mat_inv(T), 'T.I', True)

        beta_mean = np.array(beta_mean).flatten()
        beta_sample = np.random.multivariate_normal(mean=beta_mean.flatten(), cov=beta_cov, size=M)  # shape(M, 2)
        beta_sample = mat_around(beta_sample)

        aver_beta = np.sum(beta_sample, axis=0, keepdims=True) / M  # shape(1, 2)
        aver_beta = mat_around(aver_beta)

        aver_beta_squ = np.sum(np.multiply(beta_sample, beta_sample), axis=0, keepdims=True) / M  # shape(1, 2)
        aver_beta_squ = mat_around(aver_beta_squ)

        aver_beta_cross = np.sum(np.prod(beta_sample, axis=1, keepdims=True), axis=0, keepdims=True) / M  # shape(1, 1)
        aver_beta_cross = mat_around(aver_beta_cross)
        # my_debug(aver_beta, 'aver_beta', True), my_debug(aver_beta_squ, 'aver_beta_squ', True)
        # my_debug(aver_beta_cross, 'aver_beta_cross', True)

        # estimate gamma
        gamma_hat = mat_inv((W.T * T) * W) * ((W.T * T) * aver_beta.T)  # shape(2 * (q + 1), 1)
        gamma_hat = mat_around(gamma_hat)

        # estimate sigma
        sigma_squ_hat = 0
        for i in range(N):
            head = Y[i, 0] * Y[i, 0] - 2 * Y[i, 0] * aver_beta[0, 0] - 2 * X[i, 1] * Y[i, 0] * aver_beta[0, 1]
            tail = aver_beta_squ[0, 0] + 2 * X[i, 1] * aver_beta_cross.item() + X[i, 1] * aver_beta_squ[0, 1]
            sigma_squ_hat += (head + tail)
        sigma_squ_hat = sigma_squ_hat / N
        # print("sigma_squ_hat", sigma_squ_hat)

        # estimate T
        result = W * gamma_hat  # shape(2, 1)
        result = mat_around(result)

        tau00 = aver_beta_squ[0, 0] - 2 * aver_beta[0, 0] * result[0, 0] + result[0, 0] * result[0, 0]
        tau01 = (aver_beta_cross.item() - aver_beta[0, 0] * result[1, 0]
                 - aver_beta[0, 1] * result[0, 0] + result[0, 0] * result[1, 0])
        tau10 = tau01
        tau11 = aver_beta_squ[0, 1] - 2 * aver_beta[0, 1] * result[1, 0] + result[1, 0] * result[1, 0]

        T_hat = np.array([tau00, tau01, tau10, tau11]).reshape(2, 2)
        T_hat = mat_around(T_hat)
        # print(T_hat)

        gamma, sigma_squ, T = gamma_hat, sigma_squ_hat, T_hat

    # print(gamma, sigma_squ, T)
    return gamma, sigma_squ, T


def append_ones(x, w):
    """
    添加常数列，在 x, w 之前添加一列 1.
    """
    # TODO: judge W's shape is (q, ) or (q, 1)
    X = np.c_[np.ones(x.shape[0]), x]
    w = np.hstack((1, w))
    W = np.vstack((np.hstack((w, np.zeros(w.shape[0]))), np.hstack((np.zeros(w.shape[0]), w))))
    return X, W


class HLMModel(Model):
    def __init__(self, **kwargs):
        """
        gamma: matrix, 可选，第二层固定效应系数的 初始值;
                假设第二层固定效应参数有 q 个，则 gamma 为 (2 * (q + 1), 1) 的矩阵。
        sigma_squ: scalar, 第一层的误差向量的 方差的 初始值，一般假设第一层误差 r ~ N(0, sigma_squ)。
        T: matrix, 第二层误差向量的协方差矩阵的 初始值, 一般假设第二层误差 mu ~ N(0, T)。
        """
        self.gamma = None
        self.sigma_squ = None
        self.T = None

    def fit(self, **kwargs):
        """
        trainX: matrix, 必须，第一层的自变量矩阵。
        trainW: matrix, 必须，第二层的固定效应参数矩阵。
        trainY: matrix, 必须，第一层的因变量矩阵。
        iters: int, 可选，最大迭代次数。

        """
        assert "trainX" in kwargs.keys()
        assert "trainW" in kwargs.keys()
        assert "trainY" in kwargs.keys()
        iters = 24
        if "iters" in kwargs.keys():
            iters = kwargs["iters"]

        x = kwargs["trainX"]
        w = kwargs["trainW"]
        y = kwargs["trainY"]

        X, W = append_ones(x, w)
        Y = np.array(y).reshape(y.shape[0], 1)

        self.gamma, self.sigma_squ, self.T = em_hlm(W, X, Y, iters)

    def predict(self, **kwargs):
        assert "predictX" in kwargs.keys()
        assert "predictW" in kwargs.keys()

        x = kwargs["predictX"]
        w = kwargs["predictW"]

        X, W = append_ones(x, w)

        N = X.shape[0]
        p = X.shape[1]
        q = W.shape[1]

        gamma, sigma_squ, T = self.gamma, self.sigma_squ, self.T

        mu = np.random.multivariate_normal(mean=np.zeros(p), cov=T, size=M)
        mu = np.sum(mu, axis=0, keepdims=True)

        beta = np.dot(W, gamma) + mu.T
        # print(sigma_squ)
        # print(beta)

        r = np.random.normal(0, np.sqrt(sigma_squ), N)  # shape(N, )
        # print(r)

        predictY = np.dot(X, beta).flatten() + r
        return predictY


if __name__ == "__main__":
    hlm_model = HLMModel()
    hlm_dataloader = HLMDataLoader()

    datapath = '../data/hlm/'
    '''
    trainW, trainX, trainY = hlm_dataloader.loadTrainData(
        erosion_datapath=datapath + "fake_erosion_data.xlsx",
        soil_datapath=datapath + "fake_soil_data.xlsx"
    )
    testW, testX, trainY = hlm_dataloader.loadTestData(
        erosion_datapath=datapath + "fake_erosion_data.xlsx",
        soil_datapath=datapath + "fake_soil_data.xlsx"
    )
    '''

    trainW, trainX, trainY = hlm_dataloader.loadTrainData()
    testW, testX, trainY = hlm_dataloader.loadTestData()
    hlm_model.fit(trainW=trainW, trainX=trainX, trainY=trainY)
    predictY = hlm_model.predict(predictW=trainW, predictX=trainX)

    # print(trainW)
    # print(trainX)
    # print(trainY)
    print(predictY)
    print(mean_squared_error(trainY, predictY))  # mse 0.947
