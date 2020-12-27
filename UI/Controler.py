from tkinter import IntVar

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

        self.MachineLearningModel = {}  # 机器学习模型

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
        # 对第一个选项设置为选定状态
        selectFrame = list(self.view.main_left_chooseStepList.text2LabelDic.values())[1]
        selectFrame.setSelect()

        # 改变combobox选择内容
        self.view.main_right_chooseBox['values'] = self.view.main_right_chooseBox.setValues(
            self.model.loadAllModelsByGroup(self.curModelGroup))
        self.view.main_right_chooseBox.setCurrent(0)
        self.chooseModel()
        # 设置事件绑定
        self.view.main_right_chooseBox.bind(lambda event, C: C.chooseModel(), self)

        # 选择模型的步骤
        self.view.main_left_chooseStepList

    def chooseModel(self, event=None):
        """ 选择模型,在选择模型时触发
        :return:
        """
        # 读取相应参数
        curModel = self.view.main_right_chooseBox.getCurModel()
        parameterDict = self.model.loadModelParameters(curModel)
        parameterList = None
        if parameterDict == None:
            parameterList = [("No Parameter", 0)]
        else:
            parameterList = list(parameterDict.values())
        # 对于新的模型,重新布局模型页面
        self.view.main_right_parameterBox.repack(parameterList)

    def chooseStep(self, step: str):
        """ 修改步骤
        :param step: 配置模型 训练模型 ...
        :return:
        """
        pass

    def setModelParameters(self):
        """ 配置模型参数
        :return:
        """
        pass

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
    def reLayoutUI(controler: Controler, var, modelNameList, modelConfigs):
        controler.clearModelView()
        model = list(controler.ModelNameList.keys())[var.get()]
        # controler.layoutModel(model)
        pass

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
        controler.view.main_right_chooseBox.setValues(models)
        controler.view.main_right_chooseBox.setCurrent(0)
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

        }

        if text == "模型选择":
            # C.ChooseModel()
            pass
        print(text)
        return text


if __name__ == '__main__':
    model = Modeler()
    view = Viewer()
    controler = Controler(model, view)
    # controler.layoutModel()
    controler.view.run()
