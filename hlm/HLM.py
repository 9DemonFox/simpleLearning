import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from data.hlm.dataloader import HLMDataLoader
from model import Model

# global params:
SEED = 134
DECIMALS = 5
PHI = 0.000001

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
    # X = np.around(X, decimals=DECIMALS)
    return np.linalg.inv(X)


def mat_around(X):
    """
    around the matrix values.
    """
    return np.around(X, decimals=DECIMALS)


def check_precision(ge, sse, te):
    """
    :params: (ge -> gamma error), (sse -> sigma_squ), (te -> T error)
    ge and te is matrix, sse is a scalar.

    check error less than threshold or not.
    return True means yes
    """
    if np.all(ge < PHI) or sse < PHI or np.all(te < PHI):
        return True
    else:
        return False


def init_params(p, q):
    """ 随机初始的 gamma, sigma_squ, T.
    :param p: shape of T
    :param q: shape of gamma
    """
    np.random.seed(1)
    gamma = np.mat(np.random.uniform(0, 1, (q, 1)))
    np.random.seed(1)
    sigma_squ = np.random.uniform(0, 1)  # scalar
    np.random.seed(1)

    # generate positive definite mat
    A = np.random.rand(p, p)
    B = np.dot(A, A.transpose())
    C = B + B.T
    T = np.mat(C) / 10
    T[0, 1] = 0
    T[1, 0] = 0

    # T = np.mat(np.random.uniform(0, 1, (p, p)))
    return gamma, sigma_squ, T


def em_hlm3(W, X, Y, iters):
    X, W, Y = np.mat(X), np.mat(W), np.mat(Y)
    p, q, N = X.shape[1], W.shape[1], Y.shape[0]
    gamma, sigma_squ, T = init_params(p, q)

    for i in range(iters):
        # my_debug(mat_inv(T), "T.I", True)
        # print(sigma_squ)
        # my_debug(np.multiply(sigma_squ, mat_inv(T)), "result", True)
        # break

        '''estimate mu (its mean and cov)'''
        C = (X.T * X) + np.multiply(sigma_squ, mat_inv(T))  # shape(2, 2)
        mu_hat = mat_inv(C) * X.T * (Y - (X * W) * gamma)  # shape(2, 1)

        '''estimate gamma'''
        tmp = (X * W).T  # shape(q, N)
        gah_head = mat_inv(tmp * X * W)  # shape(q, q)
        gah_tail = tmp * Y - (tmp * X * mu_hat)  # shape(q, 1)
        gamma_hat = gah_head * gah_tail  # shape(q, 1)

        '''estimate T'''
        T_hat = mu_hat * mu_hat.T + np.multiply(sigma_squ, mat_inv(C))

        '''estimate sigma_squ'''
        ssh_head = (Y - X * W * gamma_hat - X * mu_hat).T * (Y - X * W * gamma_hat - X * mu_hat)
        ssh_tail = sigma_squ * np.trace(mat_inv(C) * X.T * X)
        sigma_squ_hat = (ssh_head + ssh_tail) / N

        if check_precision((gamma - gamma_hat),
                           (sigma_squ - sigma_squ_hat),
                           (T - T_hat)):
            print("break cue to threshold.")
            break

        gamma, sigma_squ, T = gamma_hat, sigma_squ_hat, T_hat

    beta_head = np.dot(X.T, X) + np.multiply(sigma_squ, mat_inv(T))
    beta_tail = np.dot(X.T, Y) + np.dot(np.multiply(sigma_squ, mat_inv(T)), np.dot(W, gamma))
    beta_hat = np.dot(mat_inv(beta_head), beta_tail)

    return gamma, sigma_squ, T, beta_hat


"""
def em_hlm2(W, X, Y, iters):
    X, W, Y = np.mat(X), np.mat(W), np.mat(Y)
    p, q, N = X.shape[1], W.shape[1], Y.shape[0]
    # print(p, q, N)
    gamma, sigma_squ, T = init_params(p, q)

    for i in range(iters):
        '''sample mu'''
        mu_mean = mat_inv(X.T * X + sigma_squ * mat_inv(T)) * X.T * (Y - (X * W) * gamma)  # shape(2, 1)
        mu_cov = sigma_squ * mat_inv(X.T * X + sigma_squ * mat_inv(T))  # shape(2, 2)
        # print(mu_mean)
        # print(mu_cov)

        mu_mean = np.array(mu_mean).flatten()
        mu_sample = np.random.multivariate_normal(mean=mu_mean.flatten(), cov=mu_cov, size=M)  # shape(M, 2)

        '''estimate gamma'''
        Lambda = T + sigma_squ * mat_inv(X.T * X)  # shape(2, 2)
        beta = mat_inv(X.T * X) * X.T * Y  # shape(2, 1)
        gamma_hat = mat_inv(W.T * Lambda * W) * (W.T * Lambda * beta)  # shape(2q, 1)

        '''estimate T'''
        T_hat = np.zeros(T.shape)
        for j in range(M):
            each_mu = mu_sample[j, :].reshape(mu_sample.shape[1], -1)  # shape(2, 1)
            T_hat += (each_mu * each_mu.T)
        T_hat = T_hat / M

        '''estimate sigma_squ'''
        mu_sum = np.sum(mu_sample, axis=0, keepdims=True)  # shape(1, 2)
        # print(type(mu_sum), mu_sum.shape)
        tmp_head = T_hat - (mu_sum.T * mu_sum) / (M * M)  # shape(2, 2)
        sig_head = np.trace(X * tmp_head * X.T)

        sig_tail = 0
        for j in range(M):
            each_mu = mu_sample[j, :].reshape(mu_sample.shape[1], -1)  # shape(2, 1)
            tmp_tail = Y - X * W * gamma_hat - X * each_mu  # shape(N, 1)
            sig_tail += (tmp_tail.T * tmp_tail).item()
        sig_tail = sig_tail / (M * M)

        sigma_squ_hat = (sig_head + sig_tail) / N
        # sigma_squ_hat = (sig_head + sig_tail)

        if check_precision(gamma_hat - gamma, sigma_squ_hat - sigma_squ, T_hat - T):
            break

        gamma, sigma_squ, T = gamma_hat, sigma_squ_hat, T_hat

    return gamma, sigma_squ, T
"""


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
        my_debug(mat_inv(T), "T.I", True)
        beta_mean = W * gamma + mat_inv(X.T * X + sigma_squ * mat_inv(T)) * X.T * (Y - (X * W) * gamma)  # shape(2, 1)
        beta_mean = mat_around(beta_mean)

        beta_cov = sigma_squ * mat_inv(X.T * X + sigma_squ * mat_inv(T))  # shape(2, 2)
        # beta_cov = mat_around(beta_cov)
        # my_debug(beta_cov, 'beta_cov', True), my_debug(mat_inv(T), 'T.I', True)
        my_debug(beta_cov, 'beta_cov', True)

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
        # my_debug(mat_inv((W.T * mat_inv(T)) * W), "gamma_head", True)
        gamma_hat = mat_inv((W.T * mat_inv(T)) * W) * ((W.T * mat_inv(T)) * aver_beta.T)  # shape(2 * (q + 1), 1)
        gamma_hat = mat_around(gamma_hat)

        # estimate sigma
        sigma_res = (Y - X * aver_beta.T).T * (Y - X * aver_beta.T)  # shape(N, 1)
        sigma_squ_hat = np.sum(sigma_res).item() / N
        """
        sigma_squ_hat = 0
        for i in range(N):
            head = Y[i, 0] * Y[i, 0] - 2 * Y[i, 0] * aver_beta[0, 0] - 2 * X[i, 1] * Y[i, 0] * aver_beta[0, 1]
            tail = aver_beta_squ[0, 0] + 2 * X[i, 1] * aver_beta_cross.item() + X[i, 1] * aver_beta_squ[0, 1]
            sigma_squ_hat += (head + tail)
        sigma_squ_hat = sigma_squ_hat / N
        # print("sigma_squ_hat", sigma_squ_hat)
        """

        # estimate T
        T_hat = (aver_beta.T - W * gamma) * (aver_beta.T - W * gamma).T
        # T_hat = (aver_beta.T - W * gamma_hat) * (aver_beta.T - W * gamma_hat).T
        """
        # result = W * gamma
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
        """

        gamma, sigma_squ, T = gamma_hat, sigma_squ_hat, T_hat

    # print(gamma, sigma_squ, T)
    return gamma, sigma_squ, T


def append_ones(x, w):
    """
    添加常数列，在 x, w 之前添加一列 1.
    """
    # TODO: judge W's shape is (q, ) or (q, 1)
    # X = np.c_[np.ones(x.shape[0]), x]
    # w = np.hstack((1, w))
    # W = np.vstack((np.hstack((w, np.zeros(w.shape[0]))), np.hstack((np.zeros(w.shape[0]), w))))
    X = np.c_[np.ones(x.shape[0]), x]
    w = np.hstack((np.ones((w.shape[0], 1)), w))
    W = np.vstack((np.hstack((w, np.zeros((1, w.shape[1])))), np.hstack((np.zeros((1, w.shape[1])), w))))
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
        self.beta = None

    def fit(self, **kwargs):
        """
        trainX: matrix, 必须，第一层的自变量矩阵。
        trainW: matrix, 必须，第二层的固定效应参数矩阵。
        trainY: matrix, 必须，第一层的因变量矩阵。
        iters: int, 可选，最大迭代次数。

        """
        assert "trainX" in kwargs.keys()
        assert "trainY" in kwargs.keys()
        # iters = 24
        iters = 500
        if "iters" in kwargs.keys():
            iters = kwargs["iters"]

        x, w = kwargs["trainX"]
        y = kwargs["trainY"]

        X, W = append_ones(x, w)
        Y = np.array(y).reshape(y.shape[0], 1)

        # self.gamma, self.sigma_squ, self.T = em_hlm(W, X, Y, iters)
        self.gamma, self.sigma_squ, self.T, self.beta = em_hlm3(W, X, Y, iters)
        return self.gamma, self.sigma_squ, self.T, self.beta

    def predict(self, **kwargs):
        assert "predictX" in kwargs.keys()
        assert "predictW" in kwargs.keys()

        x = kwargs["predictX"]
        w = kwargs["predictW"]

        X, W = append_ones(x, w)

        N = X.shape[0]
        p = X.shape[1]
        q = W.shape[1]

        gamma, sigma_squ, T, beta = self.gamma, self.sigma_squ, self.T, self.beta

        # mu = np.random.multivariate_normal(mean=np.zeros(p), cov=T, size=1)  # shape(1, 2)
        # mu = np.random.multivariate_normal(mean=np.zeros(p), cov=T, size=M)
        # mu = np.sum(mu, axis=0, keepdims=True)
        # mu = np.sum(mu, axis=0, keepdims=True) / M


        # beta = np.dot(W, gamma) + mu.T
        print("sigma_squ = ", sigma_squ)
        # print(beta)

        r = np.random.normal(0, np.sqrt(sigma_squ), N)  # shape(N, )
        # print(r)

        predictY = np.dot(X, beta).flatten() + r
        return predictY

    def fitForUI(self, **kwargs):
        returnDic = {
            "第一层模型的系数": None,
            "第二层模型的系数": None,
            "第一层随机误差的方差": None,
            "第二层随机误差的方差": None
        }
        assert "trainX" in kwargs.keys()
        assert "trainY" in kwargs.keys()
        gamma, sigma_squ, T, beta = self.fit(**kwargs)

        returnDic["第一层模型的系数"] = str(beta)
        returnDic["第二层模型的系数"] = str(gamma)
        returnDic["第一层随机误差的方差"] = str(sigma_squ)
        returnDic["第二层随机误差的方差"] = str(T)
        return returnDic

    def testForUI(self, **kwargs):
        returnDic = {
            "mean_squared_error": None,
            "mean_absolute_error": None
        }
        assert "testX" in kwargs.keys()
        assert "testY" in kwargs.keys()
        (testX, testW), testY = kwargs.get("testX"), kwargs.get("testY")
        predictResult = self.predict(predictW=testW, predictX=testX)
        print("predictResult = ", predictResult)
        mse = mean_squared_error(predictResult, testY)
        mae = mean_absolute_error(predictResult, testY)
        returnDic["mean_squared_error"] = str(mse)
        returnDic["mean_absolute_error"] = str(mae)
        return returnDic

    def predictForUI(self, **kwargs):
        returnDic = {
            "predict_result": None
        }
        assert "predictX" in kwargs.keys()
        predictX, predictW = kwargs.get("predictX")
        predictY = self.predict(predictW=predictW, predictX=predictX)
        # print("predictY = ", predictY)
        returnDic["predict_result"] = str(predictY)
        return returnDic


if __name__ == "__main__":
    hlm_model = HLMModel()
    hlm_dataloader = HLMDataLoader()

    dirname = '../data/hlm/'
    # train_path = dirname + "hlm_train_data.xlsx"
    train_path = dirname + "train_erosion_data.xlsx"
    test_path = dirname + "test_erosion_data.xlsx"
    # predict_path = dirname + "hlm_test_data.xlsx"
    predict_path = dirname + "predict_erosion_data.xlsx"

    """
    (trainX, trainW), trainY = hlm_dataloader.loadTrainData(train_path=train_path)
    X, W = append_ones(trainX, trainW)
    Y = np.array(trainY).reshape(trainY.shape[0], 1)
    em_hlm2(W, X, Y, 5)
    """

    (trainX, trainW), trainY = hlm_dataloader.loadTrainData(train_path=train_path)
    print(trainX.shape, trainW.shape, trainY.shape)
    (testX, testW), testY = hlm_dataloader.loadTestData(test_path=test_path)
    # print(testX.shape, testW.shape, testY.shape)
    (predictX, predictW) = hlm_dataloader.loadPredictData(predict_path=predict_path)
    # print(predictX.shape, predictW.shape)
    coef_dic = hlm_model.fitForUI(trainX=(trainX, trainW), trainY=trainY)
    print(coef_dic)
    # mean_error = hlm_model.testFotUI(testW=testW, testX=testX, testY=testY)
    # print(mean_error)
    # predictResult = hlm_model.predictForUI(predictX=(trainX, trainW))
    predictResult = hlm_model.predictForUI(predictX=(predictX, predictW))
    print(predictResult)

    # hlm_model.fit(trainW=trainW, trainX=trainX, trainY=trainY)
    # predictY = hlm_model.predict(predictW=trainW, predictX=trainX)

    # print(trainW)
    # print(trainX)
    # print(trainY)
    # print(predictY)
    # print(mean_squared_error(trainY, predictY))  # mse 0.947
