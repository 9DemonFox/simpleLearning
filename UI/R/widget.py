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
    backgroud = "whitesmoke"  # 当选中时，应该和背景一个颜色
    labelColor = "white"  # 当未选中时的颜色
    labelEnterColor = "whitesmoke"  # 鼠标进入时颜色
    selectedState = False

    def __init__(self, text, parent, backgroud="whitesmoke", frontgroud="white", justify=LEFT):
        super().__init__(master=parent, bg="green")
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
        self.label["bg"] = "white"
        self.seperator["bg"] = "blue"

    def setUnselect(self):
        self.selectedState = False
        self.label["bg"] = "whitesmoke"
        self.seperator["bg"] = "whitesmoke"

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
        if cls.seperator["bg"] == "blue":
            pass
        else:
            cls.seperator["bg"] = "white"
        cls.label["bg"] = "white"
        cls["bg"] = "white"
        pass

    @staticmethod
    def SleaveEvent(event, selectFrameDict: dict):
        text = event.widget["text"]
        cls = selectFrameDict.get(text)
        if cls.seperator["bg"] == "blue":
            pass
        else:
            cls.seperator["bg"] = "whitesmoke"
        if cls.selectedState == False:
            cls.label["bg"] = "whitesmoke"
            cls["bg"] = "whitesmoke"
