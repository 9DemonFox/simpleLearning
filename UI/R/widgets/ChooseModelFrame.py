from tkinter import *
from tkinter import ttk

from UI import R


class ChooseModelFrame(Frame):
    def __init__(self, parent, modelsList):
        super().__init__(parent, super().__init__(parent), bg=R.color.UNSelectedColor)
        Label(self, text="*", fg="red", bg=R.color.UNSelectedColor).pack(side=LEFT, pady=5)
        Label(self, text="模型选择", anchor=E, width=12, bg=R.color.UNSelectedColor,
              font=R.font.NormalWeiRuanYaHeiFont(10)).pack(
            side=LEFT, pady=5)
        self.curModel = StringVar()
        self.chooseBox = ttk.Combobox(self, width=20, textvariable=self.curModel)
        self.chooseBox.pack(anchor=W, padx=5, pady=5)
        self.chooseBox['values'] = modelsList  # 设置下拉列表的值
        self.chooseBox.current(0)  # 设置第一个为默认值

    def setValues(self, modelsList):
        self.chooseBox['values'] = modelsList  # 设置下拉列表的值

    def setCurrent(self, i: int):
        self.chooseBox.current(0)  # 设置第一个为默认值

    def getCurModel(self, engOnly=True):
        curModel = self.curModel.get()
        if engOnly:
            curModel = curModel[curModel.find(" ") + 1:]
        return curModel

    def bind(self, func, C):
        """ 为chooseBox绑定事件
        :param func: 回调函数 chooseBoxBind(lambda event, c: print(event, c.number.get()), C=sFL)
        :param C: 控制器
        :return:
        """
        self.chooseBox.bind("<<ComboboxSelected>>",
                            lambda event: func(event, C))
