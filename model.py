from abc import ABC, abstractmethod


# 所有模型都必须实现fit 和 predict方法
class Model(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, **kwargs):
        pass

    @abstractmethod
    def predict(self, **kwargs):
        pass
