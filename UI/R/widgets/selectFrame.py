from tkinter import *

from UI import R
from UI.R.widgets import VSeperator


class selectFrame(Frame):
    backgroud = R.color.BackGroudColor  # 当选中时，应该和背景一个颜色
    labelColor = R.color.UNSelectedColor  # 当未选中时的颜色
    labelEnterColor = R.color.SelectedColor  # 鼠标进入时颜色
    selectedState = False

    def __init__(self, text, parent, backgroud=R.color.BackGroudColor, frontgroud="white", justify=LEFT):
        super().__init__(master=parent)
        self.seperator = VSeperator(parent=self, width=5, bg=backgroud)
        self.label = Label(self, text=text, bg=backgroud, width=16, height=2, fg=frontgroud, justify=justify,
                           font=R.font.NormalWeiRuanYaHeiFont(10))

    def bind(self, event, func):
        """ 此处仅仅能绑定一个事件，否则会导致重复触发
        :param event:
        :param func:
        :return:
        """
        # super().bind(event, func)
        self.label.bind(event, func)

    def setSelect(self):
        self.selectedState = True
        self.label["bg"] = R.color.SelectedColor
        self.seperator["bg"] = R.color.SelectedSeperatorColor

    def setUnselect(self):
        self.selectedState = False
        self.label["bg"] = R.color.UNSelectedColor
        self.seperator["bg"] = R.color.UNSelectedColor

    def pack(self, padx, pady):
        super().pack(fill=X, padx=padx, pady=pady)
        self.seperator.pack()
        self.label.pack(fill=BOTH)

    @staticmethod
    def SclickEvent(event, selectFrames, C):
        """ 设置为选中状态
        :param event:
        :param cls:
        :param C: 控制器
        :return:
        """
        selectFrameDict = selectFrames.text2LabelDic
        text = event.widget["text"]
        cls = selectFrameDict.get(text)
        # 先将其它的
        for selectFrame in selectFrameDict.values():
            if selectFrame.selectedState == True:
                selectFrame.setUnselect()
        cls.setSelect()
        # 处理自定义事件 包含该UI的控制器
        selectFrames.eventHandler_(text, C)

    @staticmethod
    def SenterEvent(event, selectFrames):
        selectFrameDict = selectFrames.text2LabelDic
        text = event.widget["text"]
        cls = selectFrameDict.get(text)
        if cls.selectedState == False:
            cls.seperator["bg"] = R.color.SelectedColor
        cls.label["bg"] = R.color.SelectedColor

    @staticmethod
    def SleaveEvent(event, selectFrames):
        selectFrameDict = selectFrames.text2LabelDic
        text = event.widget["text"]
        cls = selectFrameDict.get(text)
        if cls.selectedState == False:
            cls.seperator["bg"] = R.color.UNSelectedColor
            cls.label["bg"] = R.color.UNSelectedColor
