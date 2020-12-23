import tkinter
from tkinter import IntVar, Frame, Label

from PIL import Image

from UI import CSS
from UI.Modeler import Modeler
from UI.R.text import title_string
from UI.Viewer import Viewer
from UI.Viewer import tkimg_resized


class Controler:
    def __init__(self, model: Modeler, view: Viewer):
        self.model = model
        self.initGlobalVaraible()
        self.layoutView()

    def layoutView(self, model=""):
        self.view = view
        # 添加菜单栏命令
        self.addCommandMenubar()
        self.addCommandTitleLabel()

    def addCommandTitleLabel(self):
        """ 选择不同模型
        :return:
        """
        for label in self.view.titleLabelList:
            label.bind("<Button-1>", lambda event: Command.changeModel(event, self))

    def initGlobalVaraible(self):
        self.SelectFlag = IntVar()
        self.ModelNameList = {}
        self.ModelConfigs = self.model.loadModelConfigs()
        self.modelViewWidgets = []  # 布局model的组件 后续会被清理

    def addCommandMenubar(self):
        """ 创建模型个list 选择模型
        :return:
        """
        models = self.model.loadModels()

        for i, m in enumerate(models):
            ModelName = m[0]
            ModelDetail = m[1]
            self.ModelNameList[ModelName] = ModelDetail
            self.view.filemenu.add_radiobutton(label=ModelName, variable=self.SelectFlag, value=i,
                                               command=lambda: Command.reLayoutUI(self, self.SelectFlag,
                                                                                  self.ModelNameList,
                                                                                  self.ModelConfigs))

    def clearModelView(self):
        for widget in self.modelViewWidgets:
            widget.destroy()

    def layoutModel(self, model="AHP"):
        """ 对模型进行布局
        :return:
        """
        topFrame = Frame(self.view.win)
        self.modelViewWidgets.append(topFrame)
        modelConfigs = self.ModelConfigs[model]
        topFrame.pack()
        for i, (k, v) in enumerate(modelConfigs.items()):
            klabel = Label(topFrame, text=str(k), **CSS.labelConfig)
            klabel.grid(row=i, column=0, sticky=tkinter.W)
            self.modelViewWidgets.append(klabel)
            vlabel = Label(topFrame, text=str(v))
            vlabel.grid(row=i, column=1, sticky=tkinter.W)
            self.modelViewWidgets.append(klabel)

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
        controler.layoutModel(model)
        pass

    @staticmethod
    def changeModel(event, controler: Controler):
        """ 改变页面
        :param event:
        :param controler:
        :return:
        """
        # 改变title部分
        modelGroup = event.widget["text"]
        # controler.view.main_top_image_label()
        controler.view.main_top_middle_label1["text"] = list(title_string.get(modelGroup).values())[0]
        controler.view.main_top_middle_label2["text"] = list(title_string.get(modelGroup).values())[1]
        img = title_string.get(modelGroup)["picture_path"]

        if isinstance(img, str):
            _img = Image.open(img)
        else:
            _img = img

        tk_img = tkimg_resized(_img, controler.view.main_top_image_label.winfo_width(),
                               controler.view.main_top_image_label.winfo_height(), keep_ratio=False)
        controler.view.main_top_image_label.config(image=tk_img)
        controler.view.main_top_image_label.image = tk_img
        controler.view.main_top_image_label.update()
        controler.view.main_top_middle_top_label["text"] = modelGroup


if __name__ == '__main__':
    model = Modeler()
    view = Viewer()
    controler = Controler(model, view)
    controler.layoutModel()
    controler.view.run()
