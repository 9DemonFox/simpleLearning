import tkinter as tk
from abc import ABC, abstractmethod
from tkinter import *

from UI import R


class baseSelectFrameListEvent(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def recall(self, text):
        pass


class selectFramesList(Frame):
    """ 一个用于选择处理事件的button 列表
        使用说明： 继承并且重写baseSelectFrameListEvent::recall 来处理自己的事件
        recall 参数为text 必须使用一个参数的匿名表达式
    """
    text2LabelDic = {}  # "模型中心" : Frame(模型中心)
    labels = []  # 记录的按键
    selectFlag = []  # 选中的标签

    def __init__(self, parent, recall, labelTextList=["  模型中心", "训练模型", "校验模型", "预测结果", "  数据中心", "数据集管理"],
                 width=120,
                 pady=10):
        """
        :param parent:
        :param eventHandler: 事件处理 需要继承baseSelectFrameListEvent 重写recall方法
        :param width: 组件宽度
        :param pady: list中label的间距
        """
        self.eventHandler_ = recall
        super().__init__(parent, width=width, pady=pady, bg=R.color.BackGroudColor)
        self.initLabelsList(labelTextList)
        self.Labelsbind()

    def isUnselectableTitle(self, text):
        """ 检查是不是不能选择的label
        :param text:
        :return:
        """
        return text.find("  ") >= 0

    def initLabelsList(self, labelTextList=["  模型中心", "训练模型", "校验模型", "预测结果", "  数据中心", "数据集管理"]):
        """ 初始化label列表
        :param labelTextList: ["  模型中心", "训练模型", "校验模型", "预测结果", "  数据中心", "数据集管理"]
                        其中定义2个空格开头的为Title，不可选取
        :return:
        """
        # 对于每个字符串创建selectFrame
        for text in labelTextList:
            self.text2LabelDic[text] = R.widgets.selectFrame(text=text, parent=self,
                                                             backgroud=R.color.UNSelectedColor,
                                                             frontgroud=R.color.LabelFontColor_Black)
        # 字体处理
        for text in labelTextList:
            if self.isUnselectableTitle(text):
                self.setLabelFont(text,
                                  {"justify": tk.LEFT, "anchor": tk.W, "font": R.font.BoldWeiRuanYaHeiFont(size=12)})

        # 布局
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
                    R.widgets.HSeperator(self, height=10, bg=R.color.UNSelectedColor).pack()
                    R.widgets.HSeperator(self, height=10, bg=R.color.SeperatorColor_BackGroud).pack()
                self.text2LabelDic.get(text).pack(**titlePack)
            else:
                self.text2LabelDic.get(text).pack(**selectPack)
        R.widgets.HSeperator(self, height=10, bg=R.color.FrameSeperatorColor_White).pack()

    def Labelsbind(self):
        """ 对每个Label绑定事件
        :return:
        """
        for text, selectFrame in self.text2LabelDic.items():
            if not self.isUnselectableTitle(text):  # 对于不是title的块绑定事件
                # 绑定事件
                selectFrame.bind("<Enter>",
                                 lambda event: R.widgets.selectFrame.SenterEvent(event, self))
                selectFrame.bind("<Leave>",
                                 lambda event: R.widgets.selectFrame.SleaveEvent(event, self))
                selectFrame.bind("<Button-1>",
                                 lambda event: R.widgets.selectFrame.SclickEvent(event, self))

    def setLabelFont(self, text, style: dict):
        """ 设置LabelFont
        :param text:
        :param style:
        :return:
        """
        frame = self.text2LabelDic.get(text)
        frame.label.config(**style)
