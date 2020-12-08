import unittest

# other package
from sklearn.metrics import mean_squared_error

# model
from ahp.AHP import AHPModel
# dataloader
from data.salp.dataLoder import SalpDataLoder
from salp.SALP import SVRModel


class SAPTestCase(unittest.TestCase):
    def testDataLoder(self):
        dataloader = SalpDataLoder()
        trainX, trainY = dataloader.loadTrainData()
        testX, testY = dataloader.loadTestData()
        # 验证数据集形状
        assert trainX.shape == (100, 100)
        assert trainY.shape == (100,)
        assert testX.shape == (100, 100)
        assert testY.shape == (100,)
        pass

    def testSVRModel(self):
        dataloader = SalpDataLoder()
        trainX, trainY = dataloader.loadTrainData()
        testX, testY = dataloader.loadTestData()
        model = SVRModel()
        model.fit(trainX=trainX, trainY=trainY)
        predictY = model.predict(predictX=testX)
        assert (mean_squared_error(testY, predictY) < 1)
        pass


class AHPTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)

    def testATPModel1(self):
        # 输入数据
        trainX = {'method': 'eigenvalue',
                  'name': '合理使用留成利润',
                  'criteria': ['调动职工积极性', '提高技术水平', '改善职工生活条件'],
                  'preferenceMatrices': {'criteria': [['1', '0.2', '0.33'],  # 准则层
                                                      ['5', '1', '3'],
                                                      ['3', '0.33', '1']],
                                         'subCriteria:调动职工积极性': [['1', '0.33'],
                                                                 ['3', '1']],
                                         'subCriteria:提高技术水平': [['1', '0.2'],
                                                                ['5', '1']],
                                         'subCriteria:改善职工生活条件': [['1', '0.5'],
                                                                  ['2', '1']],
                                         },
                  'subCriteria': {'调动职工积极性': ['奖金', '集体福利'],  # 决策层
                                  '提高技术水平': ['集体福利', '引进设备技术'],
                                  '改善职工生活条件': ['奖金', '集体福利'],
                                  }
                  }
        model = AHPModel()
        model.fit(trainX=trainX)
        result = model.predict()
        expect = """合理使用留成利润[1]
├── 提高技术水平[0.638]
│   ├── 引进设备技术[0.834]
│   └── 集体福利[0.167]
├── 改善职工生活条件[0.258]
│   ├── 奖金[0.333]
│   └── 集体福利[0.666]
└── 调动职工积极性[0.104]
    ├── 奖金[0.249]
    └── 集体福利[0.751]"""
        assert result.strip() == expect.strip()

    def testATPModel2(self):
        trainX = {'criteria': ['子准则层1', '子准则层2', '子准则层3', '子准则层4', '子准则层5'],
                  'method': 'eigenvalue',
                  'name': '画像',
                  'preferenceMatrices': {'criteria': [['1', '1', '2', '1', '2'],  # 准则层
                                                      ['1', '1', '2', '1', '1'],
                                                      ['0.5', '0.5', '1', '1', '1'],
                                                      ['1', '1', '1', '1', '2'],
                                                      ['0.5', '1', '1', '0.5', '1']],
                                         'subCriteria:子准则层1': [['1', '1', '1', '1'],
                                                               ['1', '1', '1', '1'],
                                                               ['1', '1', '1', '1'],
                                                               ['1', '1', '1', '1']],
                                         'subCriteria:子准则层2': [['1', '2'], ['0.5', '1']],
                                         'subCriteria:子准则层3': [['1']],
                                         'subCriteria:子准则层4': [['1', '3'], ['0.33', '1']],
                                         'subCriteria:子准则层5': [['1', '2', '3'],
                                                               ['0.5', '1', '2'],
                                                               ['0.33', '0.5', '1']]},
                  'subCriteria': {'子准则层1': ['兽残不合格量', '毒素不合格量', '污染物不合格量', '重金属不合格量'],  # 决策层
                                  '子准则层2': ['U1占比', '综合合格率'],
                                  '子准则层3': ['周期内预警触发次数*'],
                                  '子准则层4': ['牧场整改率', '牧场食品安全评审结果'],
                                  '子准则层5': ['主要理化指标Cpk*', '体细胞Cpk*', '微生物Cpk*']}}
        model = AHPModel()
        model.fit(trainX=trainX)
        result = model.predict()
        expect = """画像[1]
├── 子准则层1[0.253]
│   ├── 兽残不合格量[-7800234554605699.0]
│   ├── 毒素不合格量[2603080584620146.5]
│   ├── 污染物不合格量[2603080584620146.5]
│   └── 重金属不合格量[2603080584620146.5]
├── 子准则层2[0.225]
│   ├── U1占比[0.666]
│   └── 综合合格率[0.333]
├── 子准则层3[0.149]
│   └── 周期内预警触发次数*[1]
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
