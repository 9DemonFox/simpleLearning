import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

from data.hlm.dataloader import HLMDataLoader
from model import Model

# global params:
SEED = 9
DECIMALS = 5
PHI = 0.00001

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


def check_precision(e):
    """
    check change error less than threshold or not.
    return True means yes
    """
    if e < PHI:
        return True
    else:
        return False


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

        # if check_precision((gamma - gamma_hat), (sigma_squ - sigma_squ_hat), (T - T_hat)):
        #     print("First, break cue to threshold.")
        #     break
        # if check_precision((gamma_hat - gamma), (sigma_squ_hat - sigma_squ), (T_hat - T)):
        #     print("Second, break cue to threshold.")
        #     break

        gamma, sigma_squ, T = gamma_hat, sigma_squ_hat, T_hat

    beta_head = np.dot(X.T, X) + np.multiply(sigma_squ, mat_inv(T))
    beta_tail = np.dot(X.T, Y) + np.dot(np.multiply(sigma_squ, mat_inv(T)), np.dot(W, gamma))
    beta_hat = np.dot(mat_inv(beta_head), beta_tail)

    return gamma, sigma_squ, T, beta_hat


def em_hlm2(W, X, Y, iters):
    X, W, Y = np.mat(X), np.mat(W), np.mat(Y)
    p, q, N = X.shape[1], W.shape[1], Y.shape[0]
    gamma, sigma_squ, T = init_params(p, q)

    for i in range(iters):
        '''estimate mu (its mean and cov)'''
        mu_tmp = X.T * X + sigma_squ * mat_inv(T)
        mu_hat = mat_inv(mu_tmp) * X.T * (Y - (X * W) * gamma) # shape(2, 2)

        '''estimate gamma'''
        Lambda = T + sigma_squ * mat_inv(X.T * X)
        Beta = mat_inv(X.T * X) * X.T * Y
        gamma_hat = mat_inv(W.T * mat_inv(Lambda) * W) * (W.T * mat_inv(Lambda) * Beta)

        '''estimate T'''
        T_head = sigma_squ * mat_inv(X.T * X + sigma_squ * mat_inv(T))
        T_tail = mat_inv(X.T * X + sigma_squ * mat_inv(T)) * X.T * (Y - (X * W) * gamma)
        T_hat = T_head + (T_tail * T_tail.T)

        '''estimate sigma_squ'''
        ssh_head = np.trace(X.T * X * mat_inv(X.T * X + sigma_squ * mat_inv(T)) * sigma_squ)
        ssh_tail = (Y - X * W * gamma_hat - X * mu_hat).T * (Y - X * W * gamma_hat - X * mu_hat)
        sigma_squ_hat = (ssh_head + ssh_tail).item() / N

        # if check_precision((gamma - gamma_hat), (sigma_squ - sigma_squ_hat), (T - T_hat)):
        #     print("First, break cue to threshold.")
        #     break
        # if check_precision((gamma_hat - gamma), (sigma_squ_hat - sigma_squ), (T_hat - T)):
        #     print("Second, break cue to threshold.")
        #     break

        gamma, sigma_squ, T = gamma_hat, sigma_squ_hat, T_hat

    beta_head = np.dot(X.T, X) + np.multiply(sigma_squ, mat_inv(T))
    beta_tail = np.dot(X.T, Y) + np.dot(np.multiply(sigma_squ, mat_inv(T)), np.dot(W, gamma))
    beta_hat = np.dot(mat_inv(beta_head), beta_tail)
    # beta_hat = np.dot(W, gamma)
    print(beta_hat)

    return gamma, sigma_squ, T, beta_hat


def em_hlm(W, X, Y, iters):
    """ EM algorithm to estimate params, include gamma, sigma_squ, T.
    :param W: 第二层的固定效应参数矩阵。
    :param X: 第一层的自变量参数矩阵。
    :param Y: 第一层的因变量参数矩阵。
    :param iters: 最大迭代次数。

    :return: gamma, sigma_squ, T.
    """
    import time
    from UI import Controler

    X, W, Y = np.mat(X), np.mat(W), np.mat(Y)
    p, q, N = X.shape[1], W.shape[1], Y.shape[0]

    # init params
    gamma = np.zeros((q, 1))
    sigma_squ = 0.0001
    T = np.eye(2, 2)
    # T = np.array([[0.1, 0.2], [0.3, 0.1]]).reshape(2, 2)
    lm = LinearRegression()
    lm.fit(X[:, 1], Y)
    beta = np.zeros((2, 1))
    beta[0, 0], beta[1, 0] = lm.intercept_.item(), lm.coef_.item()
    III = 0
    for i in range(iters):
        '''Add progress bar'''
        time.sleep(0.24)  # 增加训练时间, 显示进度条效果
        Controler.PROGRESS_NOW = int((95 / iters) * i)

        '''estimate gamma'''
        delta = T + sigma_squ * mat_inv(X.T * X)  # shape(2, 2)

        # dd = np.diag(delta)  # Delta's diagonal
        # delta = np.diagflat(dd)  # just keep the Delta's diagonal
        delta_inv = mat_inv(delta)
        # print(delta_inv)

        gamma_head = W.T * delta_inv * W
        gamma_tail = (W.T * delta_inv) * beta

        gg = np.diag(gamma_head)
        gamma_head = np.diagflat(gg)
        # print(gamma_head, gamma_head.shape)
        gamma_hat = mat_inv(gamma_head) * gamma_tail  # shape(2q, 1)

        Kesai = T * delta_inv  # shape(2, 2)

        '''estimate beta'''
        bh_head = np.dot(Kesai, beta)
        bh_tail = np.dot((np.eye(2, 2) - Kesai), (W * gamma_hat))
        beta_hat = bh_head + bh_tail  # shape(2, 1)
        # print("bh_head.shape, bh_tail.shape", bh_head.shape, bh_tail.shape)
        beta = beta_hat

        '''estimate mu'''
        mu_head = X.T * X + sigma_squ * mat_inv(T)
        mu_tail = X.T * (Y - (X * W) * gamma_hat)
        mu_hat = mat_inv(mu_head) * mu_tail  # shape(2, 1)

        '''estimate T'''
        T_hat = mu_hat * mu_hat.T + sigma_squ * mat_inv(mu_head)  # shape(2, 2)
        Td = np.diag(T_hat)  # T_hat's diagonal
        T_hat = np.diagflat(Td)  # keep the T_hat diagonal

        '''estimate sigma_squ'''
        sshead_tmp = mat_inv(X.T * X + sigma_squ * mat_inv(T))
        sstail_tmp = Y - X * W * gamma_hat - X * mu_hat  # shape(N, 1)

        sigma_squ_head = np.trace(sigma_squ * (X.T * X) * sshead_tmp)  # scalar
        sigma_squ_tail = sstail_tmp.T * sstail_tmp  # shape(1, 1)
        # print("sstail", sigma_squ_tail)
        sigma_squ_hat = (sigma_squ_head + sigma_squ_tail.item()) / N

        ge, te = np.max(np.abs(gamma_hat - gamma)), np.max(np.abs(T_hat - T))
        sse = np.abs(np.sqrt(sigma_squ_hat) - np.sqrt(sigma_squ))
        III += 1
        if check_precision(ge) or check_precision(te) or check_precision(sse):
            # print("{} iterations, It breaks due to change a little.".format(III))
            break

        gamma, T, sigma_squ = gamma_hat, T_hat, sigma_squ_hat

    Controler.PROGRESS_NOW = 100

    return gamma, sigma_squ, T, beta


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
        iters = 50
        if "iters" in kwargs.keys():
            iters = kwargs["iters"]

        x, w = kwargs["trainX"]
        y = kwargs["trainY"]

        X, W = append_ones(x, w)
        Y = np.array(y).reshape(y.shape[0], 1)

        self.gamma, self.sigma_squ, self.T, self.beta = em_hlm(W, X, Y, iters)
        # self.gamma, self.sigma_squ, self.T, self.beta = em_hlm2(W, X, Y, iters)
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
        # print(r.shape)

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
        predictResult = self.predict(predictW=testW, predictX=testX).reshape(testY.shape[0], -1)
        # print("predictResult = ", predictResult.shape)
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
    train_path = dirname + "hlm_train_data.xlsx"
    # train_path = dirname + "train_erosion_data.xlsx"
    test_path = dirname + "test_erosion_data.xlsx"
    predict_path = dirname + "hlm_test_data.xlsx"
    # predict_path = dirname + "predict_erosion_data.xlsx"

    (trainX, trainW), trainY = hlm_dataloader.loadTrainData(train_path=train_path)
    # print(trainX.shape, trainW.shape, trainY.shape)
    (testX, testW), testY = hlm_dataloader.loadTestData(test_path=test_path)
    # print(testX.shape, testW.shape, testY.shape)
    (predictX, predictW) = hlm_dataloader.loadPredictData(predict_path=predict_path)
    # print(predictX.shape, predictW.shape)
    """
    (dhx, dhw), dhy = hlm_dataloader.loadTrainData(train_path='../data/hlm/hlm_train_data.xlsx')
    (bjx, bjw), bjy = hlm_dataloader.loadTrainData(train_path='../data/hlm/hlm_train_data_first.xlsx')
    DHW = np.expand_dims(dhw.flatten(), 0).repeat(dhx.shape[0], axis=0)
    DHX = np.hstack((dhx, DHW))
    BJW = np.expand_dims(bjw.flatten(), 0).repeat(bjx.shape[0], axis=0)
    BJX = np.hstack((bjx, BJW))
    X = np.vstack((DHX, BJX))
    Y = np.hstack((dhy, bjy)).T

    model = LinearRegression()
    model.fit(X, Y)

    predicts = model.predict(X)
    print(predicts)
    print(model.coef_)
    print(model.intercept_)
    """
    coef_dic = hlm_model.fitForUI(trainX=(trainX, trainW), trainY=trainY)
    print(coef_dic)
    # mean_error = hlm_model.testForUI(testX=(testX, testW), testY=testY)
    # print(mean_error)
    # predictResult = hlm_model.predictForUI(predictX=(trainX, trainW))
    predictResult = hlm_model.predictForUI(predictX=(predictX, predictW))
    print(predictResult)

