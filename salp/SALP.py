import numpy as np
from sklearn.model_selection import GridSearchCV
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
            self.model = GridSearchCV(SVR(), param_grid={"kernel": ("linear", 'rbf'), "C": np.logspace(-3, 3, 7),
                                                         "gamma": np.logspace(-3, 3, 7)})

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
