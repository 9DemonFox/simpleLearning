from abc import ABC, abstractmethod


# 所有模型都必须实现fit 和 predict方法
class Model(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, **kwargs):
        """
        :param kwargs: trainX trainY
        :return: 训练过程的参数，比如SALP会返回模型的参数，模型入选变量
        """
        pass

    @abstractmethod
    def predict(self, **kwargs):
        """
        :param kwargs: predictX
        :return:
        """
        pass
