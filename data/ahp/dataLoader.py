from data.dataLoader import DataLoader


class AHPDataLoader(DataLoader):
    def __init__(self):
        pass

    def loadTrainData(self, **kwargs):
        """
        :param
        :return: 加载数据
        """
        raise Exception("AHP Model does not have any train data")

    def loadTestData(self, **kwargs):
        """ ahp模型没有测试集
        :param kwargs:
        :return:
        """
        raise Exception("AHP Model does not have any test data")

    def loadPredictData(self, **kwargs):
        """
        :param predict_path 预测的输入
        :return: 加载数据
        """
        assert "predict_path" in kwargs.keys()
        with open(kwargs.get("predict_path"), "r", encoding="utf-8")as f:
            text = f.read()
        dic = eval(text)
        return dic


if __name__ == "__main__":
    a = AHPDataLoader()
    dic = a.loadPredictData(predict_path="")
    print(dic)
    a.loadTestData()
