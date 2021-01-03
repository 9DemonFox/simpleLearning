from tkinter import IntVar, RIGHT, YES, BOTH
from tkinter import StringVar, filedialog

from PIL import Image

from UI import R
from UI.Modeler import Modeler
from UI.R.text import title_string
from UI.Viewer import Viewer
from UI.Viewer import tkimg_resized


class Controler:
    def __init__(self, model: Modeler, view: Viewer):
        self.model = model
        self.view = view
        self.initGlobalVaraible()
        self.layoutView()
        self.MachineLearningModel = {
            "model": None,
            "parameters": {
            },
            "state": 0,  # 初始化、训练
            "train_data_path": StringVar(),  # 训练数据路径
            "train_result": StringVar(),  # 训练结果
            "test_data_path": StringVar(),  # 测试数据路径
            "predict_data_path": StringVar(),  # 预测数据路径
            "predict_result": StringVar()
        }  # 机器学习模型

        self.initState()
        # 绑定选择模型

    def initState(self):
        """ 初始化状态
        :return:
        """
        # 初试化左边步骤选择
        self.view.main_left_chooseStepList = R.widgets.selectFramesList(self.view.main_left_frame,
                                                                        labelTextList=self.model.loadMainleftFrameTextList(),
                                                                        pady=0)
        self.view.main_left_chooseStepList.pack()
        self.view.main_left_chooseStepList.Labelsbind(self)  # 绑定当前Controler
        self.view.main_left_chooseStepList.bind(
            lambda text, self: Command.navigateBar(text, self))  # 返回文本，根据文本内容控制当前的状态

        # 对第一个选项【配置模型】设置为选定状态
        selectFrame = list(self.view.main_left_chooseStepList.text2LabelDic.values())[1]
        selectFrame.setSelect()

        # 设置main_right 为模型配置页面
        self.view.main_right_1(self.view.main_frame).pack(side=RIGHT, expand=YES, fill=BOTH)
        # 改变combobox选择内容
        self.view.main_right_frame_1_chooseBox['values'] = self.view.main_right_frame_1_chooseBox.setValues(
            self.model.loadAllModelsByGroup(self.curModelGroup))
        self.view.main_right_frame_1_chooseBox.setCurrent(0)
        self.chooseModel()
        # 设置事件绑定
        self.view.main_right_frame_1_chooseBox.bind(lambda event, C: C.chooseModel(), self)

        # 选择模型的步骤
        self.view.main_left_chooseStepList

        # 初始化其它步骤时的元素
        # step2 训练模型
        self.view.main_right_2(self.view.main_frame)
        self.view.main_right_frame_2_btnPath.bind("<Button-1>",
                                                  lambda event: MainRightCommand.chooseInputData(event, "训练文件", self))

        self.view.main_right_frame_2_pathEntry.config(textvariable=self.MachineLearningModel["train_data_path"])
        self.view.main_right_frame_2_btnTrain.bind("<Button-1>",
                                                   lambda event: MainRightCommand.train(event, self))
        self.view.main_right_frame_2_txtResult.config(textvariable=self.MachineLearningModel["train_result"])

        # step4 模型数据
        self.view.main_right_4(self.view.main_frame)
        self.view.main_right_frame_4_btnPath.bind("<Button-1>",
                                                  lambda event: MainRightCommand.chooseInputData(event, "预测文件", self))
        self.view.main_right_frame_4_pathEntry.config(textvariable=self.MachineLearningModel["predict_data_path"])
        self.view.main_right_frame_4_btnPredict.bind("<Button-1>",
                                                     lambda event: MainRightCommand.predict(event, self))
        self.view.main_right_frame_4_txtResult.config(textvariable=self.MachineLearningModel["predict_result"])

    def chooseModel(self, event=None):
        """ 在选择模型时触发
        :return:
        """
        # 读取相应参数
        curModel = self.view.main_right_frame_1_chooseBox.getCurModel()

        parameterDict = self.model.loadModelParameters(curModel)
        if parameterDict == None:
            parameterList = [("No Parameter", 0)]
        else:
            parameterList = list(parameterDict.items())

        # 模型配置全局变量
        self.MachineLearningModel["parameters"] = parameterList
        self.MachineLearningModel["model"] = curModel
        # 对于新的模型,重新布局模型页面
        self.view.main_right_frame_1_parameterBox.repack(parameterList)
        # 将配置模型按键绑定并且重新触发
        self.view.main_right_frame_1_parameterBox.confirmButton.bind("<Button-1>",
                                                                     lambda event: MainRightCommand.config(event, self))
        # 触发配置事件
        MainRightCommand.config(None, self)

    def layoutView(self, modelGroup="回归模型"):
        # 添加菜单栏命令
        self.addCommandMenubar()
        self.addCommandTitleLabel()

    def addCommandTitleLabel(self):
        """ 选择不同模型
        :return:
        """
        for label in self.view.titleLabelList:
            label.bind("<Button-1>", lambda event: Command.changeModelType(event, self))
        self.view.titleLabelList[0]["fg"] = "red"

    def initGlobalVaraible(self):
        self.SelectFlag = IntVar()
        self.curModelGroup = self.model.loadAllModelsGroup()[0]
        self.ModelNameList = {}
        self.ModelConfigs = self.model.loadModelConfigs()
        self.modelViewWidgets = []  # 布局model的组件 后续会被清理

    def addCommandMenubar(self):
        """ 创建模型个list 选择模型
        :return:
        """
        models = self.model.loadModels()

    def clearModelView(self):
        for widget in self.modelViewWidgets:
            widget.destroy()

    def clearLayoutModel(self):
        """ 清理模型布局
        :return:
        """
        pass


class Command:
    def __init__(self):
        pass

    @staticmethod
    def testCommand():
        print("test")

    @staticmethod
    def printVars(var, ModelNameList):
        """ 通过控制该变量选择模型
        :param var:
        :return:
        """
        print(var.get(), )

    @staticmethod
    def changeModel(event, controlaer: Controler):
        print(event)

    @staticmethod
    def changeModelType(event, controler: Controler):
        """ 改变模型类型页面时触发
        :param event:
        :param controler:
        :return:
        """
        # 改变title部分
        modelGroup = event.widget["text"]

        # 改变选择Label颜色
        for label in controler.view.titleLabelList:
            label["fg"] = R.color.UNSelectedColor
        label = event.widget
        label["fg"] = "red"
        # 设置图片
        # controler.view.main_top_image_label()
        controler.view.main_top_middle_label1["text"] = list(title_string.get(modelGroup).values())[0]
        controler.view.main_top_middle_label2["text"] = list(title_string.get(modelGroup).values())[1]
        img = title_string.get(modelGroup)["picture_path"]
        _img = Image.open(img)
        tk_img = tkimg_resized(_img, controler.view.main_top_image_label.winfo_width(),
                               controler.view.main_top_image_label.winfo_height(), keep_ratio=False)
        controler.view.main_top_image_label.config(image=tk_img)
        controler.view.main_top_image_label.image = tk_img
        controler.view.main_top_image_label.update()
        controler.view.main_top_middle_top_label["text"] = modelGroup

        # 改变combobox选择内容
        models = controler.model.loadAllModelsByGroup(modelGroup)
        controler.view.main_right_frame_1_chooseBox.setValues(models)
        controler.view.main_right_frame_1_chooseBox.setCurrent(0)
        # 触发更改参数选择
        controler.chooseModel()

    @staticmethod
    def navigateBar(text, C: Controler):
        """ 左边导航栏变化
        :param text: Label上的标签
        :param C:  控制器 Controler
        :return:
        """
        # 所选步骤对应于右边的Frame
        step2MainRightFrame = {
            "配置模型": C.view.main_right_frame_1,
            "训练模型": C.view.main_right_frame_2,
            "预测结果": C.view.main_right_frame_4
        }

        frame = step2MainRightFrame.get(text)
        for f in step2MainRightFrame.values():
            f.pack_forget()
        frame.pack(side=RIGHT, expand=YES, fill=BOTH)
        if text == "配置模型":
            MainRightCommand.config(None, C)
        return text


class MainRightCommand:
    @staticmethod
    def chooseInputData(event, dataType: str, C: Controler):
        """
        :param event:
        :param dataType: 训练文件 测试文件 预测文件
        :param C:
        :return:
        """
        filePath = filedialog.askopenfilename()
        if dataType == "预测文件":
            if (filePath != ''):
                C.MachineLearningModel["predict_data_path"].set(filePath)
        elif dataType == "训练文件":
            if (filePath != ''):
                C.MachineLearningModel["train_data_path"].set(filePath)
        print(C.MachineLearningModel)

    @staticmethod
    def config(event, C: Controler):
        print("config model")
        """ 配置模型
        :param event:
        :param C:
        :return:
        """
        # 获取参数
        parameterDict = C.view.main_right_frame_1_parameterBox.getParametersDict()

        # 将字符串值转为数字
        for k, v in parameterDict.items():
            parameterDict[k] = eval(v)
        # 获取当前模型名字
        curModel = C.view.main_right_frame_1_chooseBox.getCurModel()
        # 初始化模型
        C.model.config_step_1(curModel, **parameterDict)
        print(curModel, parameterDict)
        pass

    @staticmethod
    def train(event, C: Controler):
        print("training")
        result = C.model.train_step_2(C.MachineLearningModel.get("train_data_path").get())
        print(result)
        pass

    @staticmethod
    def predict(event, C: Controler):
        """ 预测结果
        :param event:
        :param C:
        :return:
        """
        result = C.model.predict_step_4(C.MachineLearningModel.get("predict_data_path").get())
        C.MachineLearningModel.get("predict_result").set(result)


if __name__ == '__main__':
    model = Modeler()
    view = Viewer()
    controler = Controler(model, view)
    # controler.layoutModel()
    controler.view.run()
