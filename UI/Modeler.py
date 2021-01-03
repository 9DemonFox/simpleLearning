from UI.UIConfig import machineLearningModels, modelDefaultConfig, modelTypes2models, mainleftFrameTextList
from ahp.AHP import AHPModel
from data.ahp.dataLoader import AHPDataLoader
from data.gbm.dataLoader import GBMDataLoader
from data.hlm.dataloader import HLMDataLoader
from data.ibrt.dataLoader import IBRTDataLoader
from data.mert.dataLoder import MERTDataLoader
from data.reanfis.dataLoader import anfisDataLoader as RE_ANFISDataLoader
from data.rebet.dataLoder import REBETDataLoader
from data.salp.dataLoder import SALPDataLoader
from ga.ga import GAModel
from gbm.GBM import GBMModel
from hlm.HLM import HLMModel
from ibrt.ibrt import IBRTModel
from mert.mert import MERTModel
from re_anfis.re_anfis import re_anfisModel as RE_ANFISModel
from rebet.rebet import REBETModel
from salp.SALP import SVRModel, SALPModel


class Modeler:
    def __init__(self):
        allModels = [AHPModel, GAModel, GBMModel, HLMModel,
                     IBRTModel, MERTModel, RE_ANFISModel, REBETModel,
                     SALPModel, SVRModel]
        allDataLoader = [AHPDataLoader, None, GBMDataLoader, HLMDataLoader,
                         IBRTDataLoader, MERTDataLoader, RE_ANFISDataLoader, REBETDataLoader,
                         SALPDataLoader, SALPDataLoader]
        allModelsName = [str(m.__name__)[:-5] for m in allModels]
        self.name2Model = dict(zip(allModelsName, allModels))
        self.curModel = None  # 当前模型
        self.curModelName = None
        self.curDataLoader = None
        self.name2DataLoader = dict(zip(allModelsName, allDataLoader))

    def config_step_1(self, modelName, **parameters):
        """ 配置模型,从模型名字到模型类映射
        :return:
        """
        if "No Parameter" in parameters.keys():
            parameters = {}
        self.curModel = self.name2Model.get(modelName)(**parameters)  # 初始化模型
        self.curModelName = modelName

    def train_step_2(self, train_path):
        # DataLoader
        print(self.curModelName)
        self.curDataLoader = self.name2DataLoader.get(self.curModelName)()
        trainX, trainY = self.curDataLoader.loadTrainData(train_path=train_path)
        result = self.curModel.fitForUI(trainX=trainX, trainY=trainY)
        return result

    def predict_step_4(self, predict_path):
        """
        :param predict_path: 测试集数据路径
        :return:
        """
        # 选择当前模型
        self.curDataLoader = self.name2DataLoader.get(self.curModelName)()
        predictX = self.curDataLoader.loadPredictData(predict_path=predict_path)
        result = self.curModel.predict(predictX=predictX)
        return result

    def loadConfig(self, modelName):
        pass

    def loadModels(self):
        """ 加载所有模型
        :param configFile:
        :return:
        """
        models = machineLearningModels
        return models

    def loadModelConfigs(self):
        """ 加载所有模型配置
        :return:
        """
        configs = modelDefaultConfig
        return configs

    def loadAllModelsByGroup(self, type: str):
        models = modelTypes2models.get(type)
        modelsName = [item.get("model_name") for item in models]
        return modelsName

    def loadAllModelsGroup(self):
        return list(modelTypes2models.keys())

    def loadModelParameters(self, modelName):
        """ 从配置文件加载模型参数
        :return:
        """
        for modelType in self.loadAllModelsGroup():
            for model in modelTypes2models.get(modelType):
                name, abbr = model.get("model_name")
                if name == modelName or abbr == modelName:
                    return model.get("model_parameters")

    def loadMainleftFrameTextList(self):
        return mainleftFrameTextList


if __name__ == "__main__":
    dataLoader = AHPDataLoader()
    predictX = dataLoader.loadPredictData(
        predict_path=r"G:\研究生_UESTC\研究生-大数据中心\研究生组事务\深度学习的腐蚀预测研究\层次分析法\data\ahp\ahpInput.txt")
    model = AHPModel()
    print(model.predict(predictX=predictX))
