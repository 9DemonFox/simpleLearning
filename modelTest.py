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
from data.rfanfis.dataLoader import ANFISDataLoader
from data.salp.dataLoder import SALPDataLoader
from data.ga.dataLoder import GADataLoader
from ga.ga import GAModel
from gbm.GBM import GBMModel
from hlm.HLM import HLMModel
from ibrt.ibrt import IBRTModel
from mert.mert import MERTModel
from rebet.rebet import REBETModel
from rf_anfis.rf_anfis import rf_anfisModel
from salp.SALP import SVRModel, SALPModel

warnings.filterwarnings("ignore")


class REBETTestCase(unittest.TestCase):
    def testDataLoder(self):
        datapath1 = "./data/rebet/data_train.xlsx"
        datapath2 = "./data/rebet/data_test.xlsx"
        dataloder = REBETDataLoader()
        trainX, trainY = dataloder.loadTrainData(train_path=datapath1)
        testX, testY = dataloder.loadTestData(test_path=datapath2)
        # 验证数据集形状
        assert trainX.shape == (5000, 3)
        assert trainY.shape == (5000 ,)
        assert testX.shape == (500, 3)
        assert testY.shape == (500 ,)
        pass

    def testREBETModel(self):
        import numpy as np
        datapath1 = "./data/rebet/data_train.xlsx"
        datapath2 = "./data/rebet/data_test.xlsx"
        dataloder = REBETDataLoader()
        trainX, trainY = dataloder.loadTrainData(train_path=datapath1)
        testX, testY = dataloder.loadTestData(test_path=datapath2)
        n = 100
        epoch = 5
        k = 1
        model = REBETModel(n=n, epoch=epoch, k=k)
        model.fit(trainX=trainX, trainY=trainY)
        predictY = model.test(testX=testX, testY=testY)
        assert (np.mean(testY - predictY) < 5)
        pass


class MERTTestCase(unittest.TestCase):
    def testDataLoder(self):
        datapath1 = "./data/mert/data_train.xlsx"
        datapath2 = "./data/mert/data_test.xlsx"
        dataloder = MERTDataLoader()
        trainX, trainY = dataloder.loadTrainData(train_path=datapath1)
        testX, testY = dataloder.loadTestData(test_path=datapath2)
        # 验证数据集形状
        assert trainX.shape == (5000, 3)
        assert trainY.shape == (5000 ,)
        assert testX.shape == (500, 3)
        assert testY.shape == (500 ,)
        pass

    def testMERTModel(self):
        import numpy as np
        datapath1 = "./data/mert/data_train.xlsx"
        datapath2 = "./data/mert/data_test.xlsx"
        dataloder = MERTDataLoader()
        trainX, trainY = dataloder.loadTrainData(train_path=datapath1)
        testX, testY = dataloder.loadTestData(test_path=datapath2)
        n = 100
        epoch = 5
        k = 1
        model = MERTModel(n=n, epoch=epoch, k=k)
        model.fit(trainX=trainX, trainY=trainY)
        predictY = model.test(testX=testX, testY=testY)
        assert (np.mean(testY - predictY) < 5)
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

    def testDataLoader(self):
        dataloader = HLMDataLoader()
        trainW, trainX, trainY = dataloader.loadTrainData(train_path=self.train_path)
        testW, testX, testY = dataloader.loadTestData(test_path=self.test_path)
        predictW, predictX = dataloader.loadPredictData(predict_path=self.predict_path)

        assert trainW.shape == (1, 8)
        assert trainX.shape == (4, 1)
        assert trainY.shape == (4,)

        assert testW.shape == (1, 8)
        assert testX.shape == (4, 1)
        assert testY.shape == (4,)

        assert predictW.shape == (1, 8)
        assert predictX.shape == (4, 1)

    def testHLMModel(self):
        hlm_model = HLMModel()
        hlm_dataloader = HLMDataLoader()

        trainW, trainX, trainY = hlm_dataloader.loadTrainData(train_path=self.train_path)
        testW, testX, testY = hlm_dataloader.loadTestData(test_path=self.test_path)
        hlm_model.fit(trainW=trainW, trainX=trainX, trainY=trainY)
        predictY = hlm_model.predict(predictW=testW, predictX=testX)

        # assert mean_squared_error(trainY, predictY) < 1  # 0.947
        assert mean_squared_error(testY, predictY) < 3  # 0.947


class GATestCase(unittest.TestCase):
    def testGAModel(self):
        import numpy as np
        c = 0
        a = GADataLoader()
        F = a.loadPredictData(predict_path="./data/ga/F.txt")
        n = 3
        xmax = 10
        xmin = -10
        precisions = 24
        N_GENERATIONS = 50
        POP_SIZE = 200
        MUTATION_RATE = 0.005
        CROSSOVER_RATE = 0.8
        model = GAModel(c=c, n=n, xmax=xmax, xmin=xmin, precisions=precisions, N_GENERATIONS=N_GENERATIONS, 
                    POP_SIZE=POP_SIZE,MUTATION_RATE=MUTATION_RATE, CROSSOVER_RATE=CROSSOVER_RATE)
        value,x = model.predict(predictX=F)
        assert (F(x) == value)
        pass


class SALPTestCase(unittest.TestCase):
    def testDataLoder(self):
        dataloader = SALPDataLoader()
        trainX, trainY = dataloader.loadTrainData(train_path="data/salp/SALP_TRAIN_DATA.xlsx")
        testX, testY = dataloader.loadTestData(test_path="data/salp/SALP_TEST_DATA.xlsx")
        predictX = dataloader.loadPredictData(predict_path="data/salp/SALP_PREDICT_DATA.xlsx")
        # 验证数据集形状
        assert trainX.shape == (80, 100)
        assert trainY.shape == (80,)
        assert testX.shape == (20, 100)
        assert testY.shape == (20,)
        assert predictX.shape == (1, 100)
        pass

    def testSVRModel(self):
        dataloader = SALPDataLoader()
        trainX, trainY = dataloader.loadTrainData(train_path="data/salp/SALP_TRAIN_DATA.xlsx")
        testX, testY = dataloader.loadTestData(test_path="data/salp/SALP_TEST_DATA.xlsx")
        predictX = dataloader.loadPredictData(predict_path="data/salp/SALP_PREDICT_DATA.xlsx")
        model = SVRModel()
        model.fit(trainX=trainX, trainY=trainY)
        predict_test_Y = model.predict(predictX=testX)
        assert (mean_squared_error(testY, predict_test_Y) < 1)
        pass

    def testSALPModel(self):
        from data.salp.dataLoder import SALPDataLoader
        from sklearn.metrics import mean_squared_error

        dataloader = SALPDataLoader()
        trainX, trainY = dataloader.loadTrainData(train_path="data/salp/SALP_TRAIN_DATA.xlsx")
        testX, testY = dataloader.loadTestData(test_path="data/salp/SALP_TEST_DATA.xlsx")
        model = SALPModel()
        model.fit(trainX=trainX, trainY=trainY)
        predictY = model.predict(predictX=testX)
        assert mean_squared_error(predictY, testY) < 1


class IBRTTestCase(unittest.TestCase):
    def testDataLoader(self):
        dataloader = IBRTDataLoader()
        trainX, trainY = dataloader.loadTrainData(train_path="data/ibrt/IBRT_TRAIN_DATA.xlsx")
        testX, testY = dataloader.loadTestData(test_path="data/ibrt/IBRT_TEST_DATA.xlsx")
        assert trainX.shape == (48, 10)
        assert trainY.shape == (48,)
        assert testX.shape == (6, 10)
        assert testY.shape == (6,)
        pass

    def testIBRTModel(self):
        dataloader = IBRTDataLoader()
        trainX, trainY = dataloader.loadTrainData(train_path="data/ibrt/IBRT_TRAIN_DATA.xlsx")
        testX, testY = dataloader.loadTestData(test_path="data/ibrt/IBRT_TEST_DATA.xlsx")
        print('trainX:', trainX)
        print('trainY:', trainY)
        ibrt = IBRTModel(20, 0, 1.0, 2)
        ibrt.fit(trainX, trainY)
        predictY = ibrt.predict(testX)
        assert (mean_squared_error(testY, predictY) < 10)
        pass

class rf_anfisTestCase(unittest.TestCase):
    def testDataLoader(self):
        dataloader = ANFISDataLoader()
        trainX, trainY = dataloader.loadTrainData(train_path="data/rfanfis/RFANFIS_TRAIN_DATA.xlsx")
        testX, testY = dataloader.loadTestData(test_path="data/rfanfis/RFANFIS_TEST_DATA.xlsx")
        assert trainX.shape == (900, 3)
        assert testX.shape == (100, 3)

    def testrf_anfisModel(self):
        dataloader = ANFISDataLoader()
        train = dataloader.loadTrainData(train_path="data/rfanfis/RFANFIS_TRAIN_DATA.xlsx")
        test = dataloader.loadTestData(test_path="data/rfanfis/RFANFIS_TEST_DATA.xlsx")
        re_anfis = rf_anfisModel()
        re_anfis.fit(train)
        predictY = re_anfis.predict(test)
        #print(type(test))
        assert (mean_squared_error(test[1], predictY) < 10)
        pass


class AHPTestCase(unittest.TestCase):

    def testDataLoader(self):
        dataLoader = AHPDataLoader()
        dic = dataLoader.loadPredictData(predict_path="./data/ahp/ahpInput.txt")
        assert type(dic) == dict

    def testATPModel1(self):
        # 输入数据
        dataLoader = AHPDataLoader()
        predictX = dataLoader.loadPredictData(predict_path="./data/ahp/ahpInput.txt")
        model = AHPModel()
        result = model.predict(predictX=predictX)
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
        dataLoader = AHPDataLoader()
        predictX = dataLoader.loadPredictData(predict_path="./data/ahp/ahpInput2.txt")
        model = AHPModel()
        result = model.predict(predictX=predictX)
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
