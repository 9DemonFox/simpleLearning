from data.dataLoader import DataLoader


class AHPDataLoader(DataLoader):
    def __init__(self, data_path):
        self.data_path = data_path
        pass

    def loadTrainData(self, **kwargs):
        """
        :param
        :return: 加载数据
        """
        with open(self.data_path, "r", encoding="utf-8")as f:
            text = f.read()
        dic = eval(text)
        return dic

    def loadTestData(self, **kwargs):
        """ ahp模型没有测试集
        :param kwargs:
        :return:
        """
        raise Exception("AHP Model does not have any test data")


if __name__ == "__main__":
    a = AHPDataLoader("./ahpInput.txt")
    dic = a.loadTrainData()
    print(dic)
    a.loadTestData()
