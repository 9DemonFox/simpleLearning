from UI.UIConfig import machineLearningModels, modelDefaultConfig


class Modeler:
    def __init__(self):
        pass

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
        pass
