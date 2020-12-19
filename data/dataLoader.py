from abc import ABC, abstractmethod

class DataLoader(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def loadTrainData(self):
        """
        加载训练集
        :return:
        """
        pass

    @abstractmethod
    def loadTestData(self):
        """
        加载测试集
        :return:
        """
        pass
