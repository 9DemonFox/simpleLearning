import unittest
import warnings
import torch

# other package
from sklearn.metrics import mean_squared_error

from ahp.AHP import AHPModel
from data.ahp.dataLoader import AHPDataLoader
from data.gbm.dataLoader import GBMDataLoader
from data.hlm.dataloader import HLMDataLoader
from data.ibrt.dataLoader import IBRTDataLoader
from data.mert.dataLoder import MERTDataLoader
from data.rebet.dataLoder import REBETDataLoader
from data.salp.dataLoder import SalpDataLoader
from data.rfanfis.dataLoader import anfisDataLoader
from ga.ga import GAModel
from gbm.GBM import GBMModel
from hlm.HLM import HLMModel
from ibrt.ibrt import IBRTModel
from mert.mert import MERTModel
from rebet.rebet import REBETModel
from salp.SALP import SVRModel, SALPModel
from rf_anfis.rf_anfis import rf_anfisModel

warnings.filterwarnings("ignore")


class REBETTestCase(unittest.TestCase):
    def testDataLoder(self):
        dataloader = REBETDataLoader(datapath1="./data/rebet/data_train.csv", datapath2="./data/rebet/data_test.csv")
        trainX, trainY = dataloader.loadTrainData()
        testX, testY = dataloader.loadTestData()
        # 验证数据集形状
        assert trainX.shape == (5000, 3)
        assert trainY.shape == (5000, 1)
        assert testX.shape == (500, 3)
        assert testY.shape == (500, 1)
        pass

    def testREBETModel(self):
        import numpy as np
        dataloader = REBETDataLoader(datapath1="./data/rebet/data_train.csv", datapath2="./data/rebet/data_test.csv")
        trainX, trainY = dataloader.loadTrainData()
        testX, testY = dataloader.loadTestData()
        n = 100
        epoch = 5
        model = REBETModel(n=n)
        model.fit(trainX=trainX, trainY=trainY, epoch=epoch)
        predictY = model.predict(predictX=testX, predictY=testY)
        assert (np.mean(testY - predictY) < 1)
        pass


class MERTTestCase(unittest.TestCase):
    def testDataLoder(self):
        dataloader = MERTDataLoader(datapath1="./data/mert/data_train.csv", datapath2="./data/mert/data_test.csv")
        trainX, trainY = dataloader.loadTrainData()
        testX, testY = dataloader.loadTestData()
        # 验证数据集形状
        assert trainX.shape == (5000, 3)
        assert trainY.shape == (5000, 1)
        assert testX.shape == (500, 3)
        assert testY.shape == (500, 1)
        pass

    def testMERTModel(self):
        import numpy as np
        dataloader = MERTDataLoader(datapath1="./data/mert/data_train.csv", datapath2="./data/mert/data_test.csv")
        trainX, trainY = dataloader.loadTrainData()
        testX, testY = dataloader.loadTestData()
        n = 100
        epoch = 5
        model = MERTModel(n=n)
        model.fit(trainX=trainX, trainY=trainY, epoch=epoch)
        predictY = model.predict(predictX=testX, predictY=testY)
        assert (np.mean(testY - predictY) < 1)
        pass


class GBMTestCase(unittest.TestCase):
    train_path = "data/gbm/gbm_train_data.xlsx"
    test_path = "data/gbm/gbm_test_data.xlsx"
    predict_path = "data/gbm/gbm_predict_data.xlsx"

    def testDataLoader(self):
        dataloader = GBMDataLoader()
        trainX, trainY = dataloader.loadTrainData(train_path=self.train_path)
        testX, testY = dataloader.loadTestData(test_path=self.test_path)
        predictX = dataloader.loadPredictData(predict_path=self.predict_path)
        assert trainX.shape == (12, 9)
        assert trainY.shape == (12,)
        assert testX.shape == (4, 9)
        assert testY.shape == (4,)
        assert predictX.shape == (2, 9)
        pass

    def testGBMModel(self):
        dataloader = GBMDataLoader()
        trainX, trainY = dataloader.loadTrainData(train_path=self.train_path)
        testX, testY = dataloader.loadTestData(test_path=self.test_path)
        params = {'n_estimators': 500,
                  'max_depth': 4,
                  'min_samples_split': 5,
                  'learning_rate': 0.01,
                  'loss': 'ls'}

        gbm_model = GBMModel(**params)
        # gbm_model = GBMModel()
        gbm_model.fit(trainX=trainX, trainY=trainY)
        predictY = gbm_model.predict(predictX=testX)
        assert (mean_squared_error(testY, predictY) < 1)
        pass


class HLMTestCase(unittest.TestCase):
    train_path = "data/hlm/train_erosion_data.xlsx"
    test_path = "data/hlm/test_erosion_data.xlsx"
    predict_path = "data/hlm/predict_erosion_data.xlsx"
    soil_path = "data/hlm/soil_data.xlsx"

    def testDataLoader(self):
        dataloader = HLMDataLoader()
        trainW, trainX, trainY = dataloader.loadTrainData(train_path=self.train_path, soil_path=self.soil_path)
        testW, testX, testY = dataloader.loadTestData(test_path=self.test_path, soil_path=self.soil_path)
        predictW, predictX = dataloader.loadPredictData(predict_path=self.predict_path, soil_path=self.soil_path)

        assert trainW.shape == (1, 8)
        assert trainX.shape == (4, 1)
        assert trainY.shape == (4,)

        assert testW.shape == (1, 8)
        assert testX.shape == (4, 1)
        assert testY.shape == (4,)

        assert predictW == (1, 8)
        assert predictX == (4, 1)

    def testHLMModel(self):
        hlm_model = HLMModel()
        hlm_dataloader = HLMDataLoader()

        trainW, trainX, trainY = hlm_dataloader.loadTrainData(train_path=self.train_path, soil_path=self.soil_path)
        testW, testX, testY = hlm_dataloader.loadTestData(test_path=self.test_path, soil_path=self.soil_path)
        hlm_model.fit(trainW=trainW, trainX=trainX, trainY=trainY)
        predictY = hlm_model.predict(predictW=testW, predictX=testX)

        # assert mean_squared_error(trainY, predictY) < 1  # 0.947
        assert mean_squared_error(testY, predictY) < 3  # 0.947


class GATestCase(unittest.TestCase):
    def testGAModel(self):
        import numpy as np
        model = GAModel()

        def F(x):
            return 3 * (1 - x[0]) ** 2 * np.exp(-(x[0] ** 2) - (x[1] + 1) ** 2) - 10 * (
                    x[0] / 5 - x[0] ** 3 - x[1] ** 5) * np.exp(-x[0] ** 2 - x[1] ** 2) - 1 / 3 ** np.exp(
                -(x[0] + 1) ** 2 - x[1] ** 2) - (x[2] - 3) ** 2

        c = 1
        F = F
        n = 3
        ranges = np.array([[-3, 3], [-3, 3], [0, 4]])
        value, x = model.fit(c=c, F=F, n=n, ranges=ranges)
        assert (F(x) == value)
        pass


class SAPTestCase(unittest.TestCase):
    def testDataLoder(self):
        dataloader = SalpDataLoader("data/salp/SALP_DATA.npy")
        trainX, trainY = dataloader.loadTrainData()
        testX, testY = dataloader.loadTestData()
        # 验证数据集形状
        assert trainX.shape == (100, 100)
        assert trainY.shape == (100,)
        assert testX.shape == (100, 100)
        assert testY.shape == (100,)
        pass

    def testSVRModel(self):
        dataloader = SalpDataLoader("data/salp/SALP_DATA.npy")
        trainX, trainY = dataloader.loadTrainData()
        testX, testY = dataloader.loadTestData()
        model = SVRModel()
        model.fit(trainX=trainX, trainY=trainY)
        predictY = model.predict(predictX=testX)
        assert (mean_squared_error(testY, predictY) < 1)
        pass

    def testSALPModel(self):
        from data.salp.dataLoder import SalpDataLoader
        from sklearn.metrics import mean_squared_error

        dataloader = SalpDataLoader("./data/salp/SALP_DATA.npy")
        trainX, trainY = dataloader.loadTrainData()
        testX, testY = dataloader.loadTestData()
        model = SALPModel()
        model.fit(trainX=trainX, trainY=trainY)
        predictY = model.predict(predictX=testX)
        # FIXME SALP模型的性能不好
        assert mean_squared_error(predictY, testY) < 100


class IBRTTestCase(unittest.TestCase):
    def testDataLoader(self):
        dataloader = IBRTDataLoader()
        trainX, trainY = dataloader.loadTrainData()
        testX, testY = dataloader.loadTestData()
        assert trainX.shape == (48, 10)
        assert trainY.shape == (48,)
        assert testX.shape == (6, 10)
        assert testY.shape == (6,)
        pass

    def testIBRTModel(self):
        dataloader = IBRTDataLoader()
        trainX, trainY = dataloader.loadTrainData()
        testX, testY = dataloader.loadTestData()
        ibrt = IBRTModel(20, 0, 1.0, 2)
        ibrt.fit(trainX, trainY)
        predictY = ibrt.predict(testX)
        assert (mean_squared_error(testY, predictY) < 10)
        pass

class rf_anfisTestCase(unittest.TestCase):
    def testDataLoader(self):
        dataloader = anfisDataLoader()
        train = dataloader.loadTrainData()
        test = dataloader.loadTestData()
        assert train.dataset.tensors[0].size() == torch.Size([900, 3])
        assert test.dataset.tensors[0].size() == torch.Size([100, 3])
        pass

    def testrf_anfisModel(self):
        dataloader = anfisDataLoader()
        train = dataloader.loadTrainData()
        test = dataloader.loadTestData()
        re_anfis = rf_anfisModel()
        re_anfis.fit(train)
        predictY = re_anfis.predict(test)
        assert (mean_squared_error(test.dataset.tensors[1], predictY) < 10)
        pass


class AHPTestCase(unittest.TestCase):

    def testDataLoader(self):
        dataLoader = AHPDataLoader("./data/ahp/ahpInput.txt")
        dic = dataLoader.loadTrainData()
        assert type(dic) == dict

    def testATPModel1(self):
        # 输入数据
        dataLoader = AHPDataLoader("./data/ahp/ahpInput.txt")
        trainX = dataLoader.loadTrainData()
        model = AHPModel()
        model.fit(trainX=trainX)
        result = model.predict()
        expect = """合理使用留成利润[1]
├── 提高技术水平[0.638]
│   ├── 引进设备技术[0.833]
│   └── 集体福利[0.167]
├── 改善职工生活条件[0.258]
│   ├── 奖金[0.333]
│   └── 集体福利[0.666]
└── 调动职工积极性[0.104]
    ├── 奖金[0.249]
    └── 集体福利[0.751]"""
        assert result.strip() == expect.strip()

    def testATPModel2(self):
        dataLoader = AHPDataLoader("./data/ahp/ahpInput2.txt")
        trainX = dataLoader.loadTrainData()
        model = AHPModel()
        model.fit(trainX=trainX)
        result = model.predict()
        expect = """画像[1]
├── 子准则层1[0.253]
│   ├── 兽残不合格量[0.25]
│   ├── 毒素不合格量[0.25]
│   ├── 污染物不合格量[0.25]
│   └── 重金属不合格量[0.25]
├── 子准则层2[0.225]
│   ├── U1占比[0.666]
│   └── 综合合格率[0.333]
├── 子准则层3[0.149]
│   └── 周期内预警触发次数*[1.0]
├── 子准则层4[0.225]
│   ├── 牧场整改率[0.751]
│   └── 牧场食品安全评审结果[0.249]
└── 子准则层5[0.149]
    ├── 主要理化指标Cpk*[0.54]
    ├── 体细胞Cpk*[0.297]
    └── 微生物Cpk*[0.163]"""
        assert result.strip() == expect.strip()


if __name__ == '__main__':
    unittest.main()
