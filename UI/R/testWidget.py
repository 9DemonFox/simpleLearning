import tkinter as tk
from tkinter import Frame

from UI import R


class Viewer:
    def __init__(self):
        self.initWindow()

    def run(self):
        self.win.mainloop()

    def initWindow(self):
        self.win = tk.Tk()
        self.win.title("材料腐蚀预测")  # 添加标题
        self.win.geometry("{}x{}+{}+{}".format(960, 640, 100, 0))
        # sF = R.widget.selectFrame("测试", self.win, backgroud="whitesmoke", frontgroud="black")
        # sF.pack()
        # # sF.setSelect()
        # sF.bind("<Button-1>", lambda event: R.widget.selectFrame.SsetSelect(event, sF))
        # sF.bind("<Enter>", lambda event: R.widget.selectFrame.SenterEvent(event, sF))
        # sF.bind("<Leave>", lambda event: R.widget.selectFrame.SleaveEvent(event, sF))
        mainFrame = Frame(self.win, bg="red")

        sFL = SelectLabelsList(mainFrame)
        sFL.initLabelsList()
        sFL.pack()
        sFL.Labelsbind()
        mainFrame.pack()


"""
    def main_left(self, parent):
        def label(frame, text, size=10, bold=False, bg="white"):
            return tk.Label(frame, text=text, bg=bg, font=_ft(size, bold))

        labels_list = []

        frame = tk.Frame(parent, width=180, bg="white")
        label(frame, "模型中心", 12, True).pack(anchor=tk.W, padx=20, pady=10)
        label(frame, "我的模型").pack(anchor=tk.W, padx=40, pady=5)

        f1 = tk.Frame(frame, bg="whitesmoke")
        v_seperator(f1, width=5, bg="blue").pack(side=tk.LEFT, fill=tk.Y)
        label(f1, "创建模型", bg="whitesmoke").pack(side=tk.LEFT, anchor=tk.W, padx=35, pady=5)
        f1.pack(fill=tk.X)

        label(frame, "训练模型").pack(anchor=tk.W, padx=40, pady=5)
        label(frame, "校验模型").pack(anchor=tk.W, padx=40, pady=5)
        label(frame, "发布模型").pack(anchor=tk.W, padx=40, pady=5)

        h_seperator(frame, 10)

        label(frame, "数据中心", 12, True).pack(anchor=tk.W, padx=20, pady=10)
        label(frame, "数据集管理").pack(anchor=tk.W, padx=40, pady=5)
        label(frame, "创建数据集").pack(anchor=tk.W, padx=40, pady=5)

        frame.propagate(False)
        return frame
"""


class SelectLabelsList(Frame):
    text2LabelDic = {}  # "模型中心" : Frame(模型中心)
    labels = []  # 记录的按键
    selectFlag = []  # 选中的标签

    def __init__(self, parent, width=120, pady=10):
        super().__init__(parent, width=width, pady=pady, bg="whitesmoke")

    def isUnselectableTitle(self, text):
        """ 检查是不是不能选择的label
        :param text:
        :return:
        """
        return text.find("  ") >= 0

    def initLabelsList(self):
        """ 初始化label列表
        :param labelTextList: 包含标签列表 ["模型中心","数据中心"]
        :return:
        """
        labelTextList = ["  模型中心", "训练模型", "校验模型", "预测结果", "  数据中心", "数据集管理"]

        # 对于每个字符串创建selectFrame
        for text in labelTextList:
            self.text2LabelDic[text] = R.widget.selectFrame(text=text, parent=self,
                                                            backgroud="whitesmoke",
                                                            frontgroud="black")

        # 不同字体处理
        self.setLabelFont("  模型中心", {"justify": tk.LEFT, "anchor": tk.W, "font": R.font.BoldWeiRuanYaHeiFont(size=12)})
        self.setLabelFont("  数据中心", {"justify": tk.LEFT, "anchor": tk.W, "font": R.font.BoldWeiRuanYaHeiFont(size=12)})

        titlePack = {
            "pady": 2,
            "padx": 0
        }
        selectPack = {
            "pady": 2,
            "padx": 0
        }

        for i, (text, selectFrame) in enumerate(self.text2LabelDic.items()):
            if self.isUnselectableTitle(text):
                if i != 0:
                    R.widget.HSeperator(self, height=10, bg="white").pack()
                self.text2LabelDic.get(text).pack(**titlePack)
            else:
                self.text2LabelDic.get(text).pack(**selectPack)
        R.widget.HSeperator(self, height=10, bg="whitesmoke").pack()

    def Labelsbind(self):
        for text, selectFrame in self.text2LabelDic.items():
            if not self.isUnselectableTitle(text):
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


if __name__ == '__main__':
    v = Viewer()
    v.run()
