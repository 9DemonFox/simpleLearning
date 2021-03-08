import unittest
import warnings

# other package
from sklearn.metrics import mean_squared_error

from ahp.AHP import AHPModel
from data.ahp.dataLoader import AHPDataLoader
from data.gbm.dataLoader import GBMDataLoader
from data.hlm.dataloader import HLMDataLoader
from data.ibrt.dataLoader import IBRTDataLoader
from data.rebet.dataLoder import REBETDataLoader
from data.rfanfis.dataLoader import ANFISDataLoader
from data.salp.dataLoder import SALPDataLoader
from gbm.GBM import GBMModel
from hlm.HLM import HLMModel
from ibrt.ibrt import IBRTModel
from rebet.rebet import REBETModel
from rf_anfis.rf_anfis import RF_ANFISModel
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
        pass

    def testREBETModel(self):
        datapath1 = "./data/rebet/data_train.xlsx"
        datapath2 = "./data/rebet/data_test.xlsx"
        dataloder = REBETDataLoader()
        trainX, trainY = dataloder.loadTrainData(train_path=datapath1)
        testX, testY = dataloder.loadTestData(test_path=datapath2)
        n = 1
        epoch = 5
        k = 1
        model = REBETModel(n=n, epoch=epoch, k=k)
        model.fit(trainX=trainX, trainY=trainY)
        model.test(testX=testX, testY=testY)
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

        gbm_model = GBMModel(**params,ga=0)
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
        (trainX, trainW), trainY = dataloader.loadTrainData(train_path=self.train_path)
        (testX, testW), testY = dataloader.loadTestData(test_path=self.test_path)
        (predictX, predictW) = dataloader.loadPredictData(predict_path=self.predict_path)

        assert trainW.shape == (1, 8)
        assert trainX.shape == (20, 1)
        assert trainY.shape == (20,)

        assert testW.shape == (1, 8)
        assert testX.shape == (4, 1)
        assert testY.shape == (4,)

        assert predictW.shape == (1, 8)
        assert predictX.shape == (4, 1)

    def testHLMModel(self):
        hlm_model = HLMModel()
        dataloader = HLMDataLoader()

        (trainX, trainW), trainY = dataloader.loadTrainData(train_path=self.train_path)
        (testX, testW), testY = dataloader.loadTestData(test_path=self.test_path)
        hlm_model.fit(trainX=(trainX, trainW), trainY=trainY)
        predictY = hlm_model.predict(predictW=testW, predictX=testX).reshape(testY.shape[0], -1)

        print(mean_squared_error(testY, predictY))  # 0.947
        assert mean_squared_error(testY, predictY) < 3  # 0.947

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
        trainX, trainY = dataloader.loadTrainData(train_path="/Users/dssa/Downloads/0305/train_data.xlsx")
        testX = dataloader.loadPredictData(predict_path="/Users/dssa/Downloads/0305/test_data.xlsx")
        print('trainX:', trainX.shape)
        print('trainY:', trainY)
        print('testX:', testX.shape)
        ibrt = IBRTModel(200, 3)
        ibrt.fit(trainX=trainX, trainY=trainY)
        predictY = ibrt.predict(predictX=testX)
        print('predictY:', predictY)
        #assert (mean_squared_error(testY, predictY) < 1)
        pass


class rf_anfisTestCase(unittest.TestCase):
    def testDataLoader(self):
        dataloader = ANFISDataLoader()
        trainX, trainY = dataloader.loadTrainData(train_path="data/rfanfis/RFANFIS_TRAIN_DATA.xlsx")
        testX, testY = dataloader.loadTestData(test_path="data/rfanfis/RFANFIS_TEST_DATA.xlsx")
        assert trainX.shape == (900, 3)
        assert testX.shape == (100, 3)

    def test_rf_anfisModel(self):
        dataloader = ANFISDataLoader()
        trainX, trainY = dataloader.loadTrainData(train_path="/Users/dssa/Downloads/0305/train_data.xlsx")
        testX = dataloader.loadPredictData(predict_path="/Users/dssa/Downloads/0305/test_data.xlsx")

        re_anfis = RF_ANFISModel(trainX,num_mfs=2, c=0.5)
        re_anfis.fit(trainX=trainX, trainY=trainY)
        predictY = re_anfis.predict(predictX= testX)
        print('predictY:',predictY)
        #assert (mean_squared_error(testY, predictY) < 10)

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
