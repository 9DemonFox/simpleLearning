from UI.UIConfig import machineLearningModels, modelDefaultConfig, modelTypes2models, mainleftFrameTextList


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

    def loadAllModelsByGroup(self, type: str):
        models = modelTypes2models.get(type)
        modelsName = [item.get("model_name") for item in models]
        return modelsName

    def loadAllModelsGroup(self):
        return list(modelTypes2models.keys())

    def loadModelParameters(self, modelName):
        """ 加载模型参数
        :return:
        """
        for modelType in self.loadAllModelsGroup():
            for model in modelTypes2models.get(modelType):
                name, abbr = model.get("model_name")
                if name == modelName or abbr == modelName:
                    return model.get("model_parameters")

    def loadMainleftFrameTextList(self):
        return mainleftFrameTextList


if __name__ == '__main__':
    m = Modeler()
    parameter = m.loadModelParameters("SALP")
    print(parameter)
