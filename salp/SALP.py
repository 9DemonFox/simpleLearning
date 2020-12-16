import numpy
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import normalize
from sklearn.svm import SVR

from model import Model


class SVRModel(Model):

    def __init__(self, **kwargs):
        """
        kernel ： string，optional（default ='rbf'）
        指定要在算法中使用的内核类型。它必须是'linear'，'poly'，'rbf'，'sigmoid'，'precomputed'或者callable之一。如果没有给出，将使用'rbf'。如果给出了callable，则它用于预先计算内核矩阵。
        degree： int，可选（默认= 3）
        多项式核函数的次数（'poly'）。被所有其他内核忽略。
        gamma ： float，optional（默认='auto'）
        'rbf'，'poly'和'sigmoid'的核系数。
        当前默认值为'auto'，它使用1 / n_features，如果gamma='scale'传递，则使用1 /（n_features * X.std（））作为gamma的值。当前默认的gamma''auto'将在版本0.22中更改为'scale'。'auto_deprecated'，'auto'的弃用版本用作默认值，表示没有传递明确的gamma值。
        coef0 ： float，optional（默认值= 0.0）
        核函数中的独立项。它只在'poly'和'sigmoid'中很重要。
        tol ： float，optional（默认值= 1e-3）
        容忍停止标准。
        C ： float，可选（默认= 1.0）
        错误术语的惩罚参数C.
        epsilon ： float，optional（默认值= 0.1）
        Epsilon在epsilon-SVR模型中。它指定了epsilon-tube，其中训练损失函数中没有惩罚与在实际值的距离epsilon内预测的点。
        收缩 ： 布尔值，可选（默认= True）
        是否使用收缩启发式。
        cache_size ： float，可选
        指定内核缓存的大小（以MB为单位）。
        详细说明 ： bool，默认值：False
        启用详细输出。请注意，此设置利用libsvm中的每进程运行时设置，如果启用，则可能无法在多线程上下文中正常运行。
        max_iter ： int，optional（默认值= -1）
        求解器内迭代的硬限制，或无限制的-1
        :param kwargs: useDefualtParameter = True 表示使用模型的默认参数，参考论文 referPaper/基于Lasso方法的碳钢土壤腐蚀率预报研究_鲁庆.pdf
        :return:
        """
        if "useDefaultParameter" in kwargs.keys() and kwargs["useDefaultParameter"] == True:
            self.model = SVR()
        else:
            self.model = GridSearchCV(SVR(), param_grid={"kernel": ("linear", 'rbf'), "C": numpy.logspace(-3, 3, 7),
                                                         "gamma": numpy.logspace(-3, 3, 7)})

    def fit(self, **kwargs):
        assert "trainX" in kwargs.keys()
        assert "trainY" in kwargs.keys()
        trainX = kwargs["trainX"]
        trainY = kwargs["trainY"]
        self.model.fit(trainX, trainY)

    def predict(self, **kwargs):
        assert "predictX" in kwargs.keys()
        predictX = kwargs["predictX"]
        return self.model.predict(predictX)


class SALPModel(Model):
    def __init__(self, alpha=0.00002, keepVariablePercent=0.25, k=10):
        """
        :param alpha:  惩罚项系数
        :param keepVariable: 选择标量排除分位数 0.25 则是去除25%的最少次数选择变量
        :param k: 贝叶斯重构样本数量
        """
        self.alpha = alpha
        self.keepPercent = keepVariablePercent
        self.k = k
        pass

    def normalXY(self, x, y):
        """标准化 均值为0 平方和为1
        :param x:
        :param y:
        :return:
        """
        normal_l2 = lambda data: normalize(data.reshape(1, -1), norm="l2").squeeze()
        y = normal_l2(y)
        x = numpy.apply_along_axis(normal_l2, axis=1, arr=x)
        return x, y

    def bayesian_bootstrap(self, X,
                           statistic,
                           n_replications,
                           resample_size,
                           low_mem=False):
        """
        :param statistic: 对于抽象，取func = lambda x: x
        :param n_replications: 重复次数
        :param resample_size:
        :param low_mem:
        :return: 重抽样数据样本和抽样数据Index
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

    def split_xy(self, xy):
        """ 分开测试集和训练集
        [100,101] =>[100,100],[100,1]
        :return:
        """
        x = xy[:, 1:]
        y = xy[:, 0]
        return x, y

    def getBayesianBootstrapReconstructData(self, x, y, n_replications):
        """ 使用bayesian_tootstrap重构数据集
        :param y:
        :param n_replications: 重构样本数量
        :return:
        """
        # 合并x,y =>[100,100],[100,1]=>[100,101]
        std_xy = numpy.insert(x, 0, values=y, axis=1)
        bayes_xys, bayes_indexs = self.bayesian_bootstrap(std_xy,
                                                          lambda x: x,
                                                          n_replications=n_replications,
                                                          resample_size=len(std_xy))
        xs = []
        ys = []
        for i in range(n_replications):
            x, y = self.split_xy(bayes_xys[i])
            xs.append(x)
            ys.append(y)
        return (xs, ys, bayes_indexs)

    def getPLSCoef(self, x, y):
        """ 获取偏最小二乘回归系数(即是ALP算法中P的含义 Pls)获取的系数
        :param x:
        :param y:
        :return:
        """
        pls = PLSRegression()
        pls.fit(x, y)
        return pls.coef_

    def getLassoCoef(self, x, y, alpha=0):
        """
        :param x:
        :param y:
        :return:
        """
        lasso = Lasso(alpha=alpha)
        lasso.fit(x, y)
        return lasso.coef_

    def adap_lasso_with_init_weight(self, x, y, initWeight="ols", alpha=0.000005):
        """
        :param x:
        :param y:
        :param initWeight: 可以选ols、pls、lasso
        :return:
        """
        Coef = None
        sampleNum = x.shape[0]  # 样本数量
        varsNum = x.shape[1]  # 变量数量(特征数量)
        if initWeight == "ols":
            Coef = self.getOlsCoef(x, y).reshape(-1)
        elif initWeight == "pls":
            Coef = self.getPLSCoef(x, y).reshape(-1)  # 获取偏最小二乘估计得到的系数
        elif initWeight == "lasso":
            Coef = self.getLassoCoef(x, y, alpha=alpha).reshape(-1)
        else:
            Coef = numpy.ones(varsNum)
        # 权重为绝对值倒数的gama次方
        # referPaper/基于数据挖掘的材料自然环境腐蚀预测研究.pdf page67 公式(4.6)
        gama = 1
        w = 1 / (abs(Coef + numpy.finfo(float).eps)) ** gama
        ws = numpy.tile(w, sampleNum).reshape([sampleNum, varsNum])  # 显式的将权重扩充到和x共维度
        # 通过变换把adaptive Lass 变为普通Lasso
        # 给定最小值，避免出现无穷值
        ws[numpy.where(ws <= 0.01 / varsNum)] = 0.01 / varsNum  # 给定最小值，避免出现无穷值

        xStar = x / ws  # 置换x为xStar（x*）
        # xStar, y = self.normalXY(xStar, y)  # 对xStar和y归一化
        alp = Lasso(alpha=0.000005)  # 通过变换x来得到相关权重
        alp.fit(xStar, y)
        return alp, w
        # 再使用Lasso

    def getALPCoef(self, x, y, initWeight="ols", alpha=0.000005):
        """ 获取ALP选取的参数
        :param x:
        :param y:
        :return:
        """
        alp, w = self.adap_lasso_with_init_weight(x, y, initWeight=initWeight, alpha=alpha)
        coef = alp.coef_ / w
        return coef

    def getOlsCoef(self, x, y):
        lasso = LinearRegression()
        lasso.fit(x, y)
        return lasso.coef_

    def voteCoef(self, coef, Vote):
        """ 对变量进行计数
        :param x:
        :param y:
        :return:
        """
        assert Vote.shape == coef.shape
        coef_index = numpy.where(coef != 0)
        Vote[coef_index] = 1 + Vote[coef_index]
        return Vote

    def fit(self, **kwargs):
        x = kwargs.get("trainX")
        y = kwargs.get("trainY")
        # 对数据进行中心化 均值为0 平方和为1
        std_x, std_y = self.normalXY(x, y)
        # 验证均值为0 平方和为1
        assert abs(numpy.square(std_y).sum() - 1) < 0.01
        assert abs(numpy.square(std_x[0]).sum() - 1) < 0.01
        assert abs(abs(std_y.mean()) < 0.1)
        # step2
        (xs, ys, bayes_indexs) = self.getBayesianBootstrapReconstructData(std_x, std_y, n_replications=self.k)
        # step3
        variable_num = x.shape[1]
        Vote = numpy.zeros(variable_num)  # 对于留下的样本计数
        for L in range(self.k):  # 对于每个贝叶斯数据集
            xL, yL = xs[L], ys[L]  # 取出当前样本
            coef = self.getALPCoef(xL, yL)
            Vote = self.voteCoef(coef, Vote)

        # print(Vote)  # 变量预选

        def getExcludeIndex(vote, percent=0.25):
            # 按照分位数方法，得出需要排除的index
            def getPercentNum(arr, p):
                # 求分位数
                arr.sort()
                return arr[int(p * len(arr))]

            v = numpy.copy(vote)
            p_num = getPercentNum(v, percent)
            return numpy.where(vote <= p_num), p_num  # 需要被排除的index,以及分位数

        def removeFeatures(arr, removeIndexs):
            arr[:, removeIndexs] = 0
            return arr

        # 将落选的变量置0
        index, pnum = getExcludeIndex(Vote)
        Xstar = removeFeatures(std_x, index)

        # 再使用Lasso获取模型
        modelEnd, _ = self.adap_lasso_with_init_weight(Xstar, std_y)

        # 得到系数
        self.model = modelEnd

    def predict(self, **kwargs):
        assert "predictX" in kwargs.keys()
        predictX = kwargs["predictX"]
        return self.model.predict(predictX)


if __name__ == "__main__" and False:
    from data.salp.dataLoder import SalpDataLoder
    from sklearn.metrics import mean_squared_error

    dataloader = SalpDataLoder("../data/salp/SALP_DATA.npy")
    trainX, trainY = dataloader.loadTrainData()
    testX, testY = dataloader.loadTestData()
    model = SVRModel()
    model.fit(trainX=trainX, trainY=trainY)
    predictY = model.predict(predictX=testX)
    assert (mean_squared_error(testY, predictY) < 1)
    pass

if __name__ == "__main__" and False:
    from data.salp.dataLoder import SalpDataLoder
    from sklearn.metrics import mean_squared_error

    dataloader = SalpDataLoder("../data/salp/SALP_DATA.npy")
    trainX, trainY = dataloader.loadTrainData()
    testX, testY = dataloader.loadTestData()
    model = SALPModel()
    # step1
    std_x, std_y = model.normalXY(trainX, trainY)
    print("验证均值为0 平方和为1:", (std_y.mean(), numpy.square(std_y).sum()),
          (std_x[0].mean(), numpy.square(std_x[0]).sum()))  # 论文中要求数据)
    # step2
    k = 10  # 重构样本数量
    (xs, ys, bayes_indexs) = model.getBayesianBootstrapReconstructData(std_x, std_y, n_replications=k)
    # step3
    d = 100  # 变量数量
    Vote = numpy.zeros(d)  # 对于留下的样本计数
    aim_v_num = []  # 目标变量计算
    for L in range(k):  # 对于每个模型
        xL, yL = xs[L], ys[L]  # 取出当前样本
        coef = model.getALPCoef(xL, yL)
        Vote = model.voteCoef(coef, Vote)
    print(Vote)


    def getExcludeIndex(vote, percent=0.25):
        # 按照分位数方法，得出需要排除的index
        def getPercentNum(arr, p):
            # 求分位数
            arr.sort()
            return arr[int(p * len(arr))]

        v = numpy.copy(vote)
        p_num = getPercentNum(v, percent)
        return numpy.where(vote <= p_num), p_num  # 需要被排除的index,以及分位数


    def removeFeatures(arr, removeIndexs):
        arr[:, removeIndexs] = 0
        return arr


    index, pnum = getExcludeIndex(Vote)
    Xstar = removeFeatures(std_x, index)
    modelEnd, _ = model.adap_lasso_with_init_weight(Xstar, std_y)
    print(modelEnd.coef_)
    # 这些index列的特征全部置0

    # 根据重构样本重新选择变量组织算法
    # 对于落选的变量，使得他们的特征为0即可

    # plt.scatter(numpy.linspace(0, 100, 100), model.getPLSCoef(xL, yL).reshape(100))
    # plt.scatter(, model.getPLSCoef(xL, yL))
    # plt.show()

if __name__ == "__main__" and True:
    from data.salp.dataLoder import SalpDataLoder
    from sklearn.metrics import mean_squared_error

    dataloader = SalpDataLoder("../data/salp/SALP_DATA.npy")
    trainX, trainY = dataloader.loadTrainData()
    testX, testY = dataloader.loadTestData()
    model = SALPModel()
    model.fit(trainX=trainX, trainY=trainY)
    predictY = model.predict(predictX=testX)
    print(mean_squared_error(predictY, testY))
    print(model.model.coef_)
