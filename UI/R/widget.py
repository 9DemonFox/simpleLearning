import tkinter as tk
from tkinter import *
from tkinter import font as tkFont

from UI import R


# f1 = tk.Frame(frame, bg="whitesmoke")
# v_seperator(f1, width=5, bg="blue").pack(side=tk.LEFT, fill=tk.Y)
# label(f1, "创建模型", bg="whitesmoke").pack(side=tk.LEFT, anchor=tk.W, padx=35, pady=5)
# f1.pack(fill=tk.X)


def _font(fname="微软雅黑", size=12, bold=tkFont.NORMAL):
    """设置字体"""
    ft = tkFont.Font(family=fname, size=size, weight=bold)
    return ft


def _ft(size=12, bold=False):
    """极简字体设置函数"""
    if bold:
        return _font(size=size, bold=tkFont.BOLD)
    else:
        return _font(size=size, bold=tkFont.NORMAL)


class VSeperator(Frame):
    """垂直分割线 , fill=tk.Y, 但如何定位不确定，直接返回对象，由容器决定 """

    def __init__(self, parent, width, bg="whitesmoke"):  # width 单位为像素值
        super().__init__(parent, width=width, bg=bg)

    def pack(self):
        super().pack(side=LEFT, fill=Y)


class HSeperator(Frame):  # height 单位为像素值
    """水平分割线
    """

    def __init__(self, parent, height, bg="whitesmoke"):  # width 单位为像素值
        super().__init__(parent, height=height, bg=bg)

    def pack(self):
        super().pack(fill=X)


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
        # self.seperator.pack()
        # self.label.pack(side=LEFT, anchor=W, padx=35, pady=5)

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
        # self.label.pack(side=LEFT, anchor=W, padx=padx, pady=pady)

    @staticmethod
    def SclickEvent(event, selectFrameDict: dict):
        """ 设置为选中状态
        :param event:
        :param cls:
        :return:
        """
        text = event.widget["text"]
        cls = selectFrameDict.get(text)
        # 先将其它的
        for selectFrame in selectFrameDict.values():
            if selectFrame.selectedState == True:
                selectFrame.setUnselect()
        cls.setSelect()

    @staticmethod
    def SenterEvent(event, selectFrameDict: dict):
        text = event.widget["text"]
        cls = selectFrameDict.get(text)
        if cls.selectedState == False:
            cls.seperator["bg"] = R.color.SelectedColor
        cls.label["bg"] = R.color.SelectedColor

    @staticmethod
    def SleaveEvent(event, selectFrameDict: dict):
        text = event.widget["text"]
        cls = selectFrameDict.get(text)
        if cls.selectedState == False:
            cls.seperator["bg"] = R.color.UNSelectedColor
            cls.label["bg"] = R.color.UNSelectedColor


class SelectLabelsList(Frame):
    text2LabelDic = {}  # "模型中心" : Frame(模型中心)
    labels = []  # 记录的按键
    selectFlag = []  # 选中的标签

    def __init__(self, parent, width=120, pady=10):
        super().__init__(parent, width=width, pady=pady, bg=R.color.BackGroudColor)

    def isUnselectableTitle(self, text):
        """ 检查是不是不能选择的label
        :param text:
        :return:
        """
        return text.find("  ") >= 0

    def initLabelsList(self):
        """ 初始化label列表
        :param labelTextList: ["  模型中心", "训练模型", "校验模型", "预测结果", "  数据中心", "数据集管理"]
                        其中定义2个空格开头的为Title，不可选取
        :return:
        """
        labelTextList = ["  模型中心", "训练模型", "校验模型", "预测结果", "  数据中心", "数据集管理"]

        # 对于每个字符串创建selectFrame
        for text in labelTextList:
            self.text2LabelDic[text] = R.widget.selectFrame(text=text, parent=self,
                                                            backgroud=R.color.UNSelectedColor,
                                                            frontgroud=R.color.LabelFontColor_Black)
        # 不同字体处理
        for text in labelTextList:
            if self.isUnselectableTitle(text):
                self.setLabelFont(text,
                                  {"justify": tk.LEFT, "anchor": tk.W, "font": R.font.BoldWeiRuanYaHeiFont(size=12)})

        titlePack = {
            "pady": 0,
            "padx": 0
        }
        selectPack = {
            "pady": 0,
            "padx": 0
        }

        for i, (text, selectFrame) in enumerate(self.text2LabelDic.items()):
            if self.isUnselectableTitle(text):
                if i != 0:
                    R.widget.HSeperator(self, height=10, bg=R.color.UNSelectedColor).pack()
                    R.widget.HSeperator(self, height=10, bg=R.color.SeperatorColor_BackGroud).pack()
                self.text2LabelDic.get(text).pack(**titlePack)
            else:
                self.text2LabelDic.get(text).pack(**selectPack)
        R.widget.HSeperator(self, height=10, bg=R.color.FrameSeperatorColor_White).pack()

    def Labelsbind(self):
        """ 对每个Label绑定事件
        :return:
        """
        for text, selectFrame in self.text2LabelDic.items():
            if not self.isUnselectableTitle(text):  # 对于不是title的块绑定事件
                # 绑定事件
                selectFrame.bind("<Enter>",
                                 lambda event: R.widget.selectFrame.SenterEvent(event, self.text2LabelDic))
                selectFrame.bind("<Leave>",
                                 lambda event: R.widget.selectFrame.SleaveEvent(event, self.text2LabelDic))
                selectFrame.bind("<Button-1>",
                                 lambda event: R.widget.selectFrame.SclickEvent(event, self.text2LabelDic))

    def setLabelFont(self, text, style: dict):
        """ 设置LabelFont
        :param text:
        :param style:
        :return:
        """
        frame = self.text2LabelDic.get(text)
        frame.label.config(**style)
