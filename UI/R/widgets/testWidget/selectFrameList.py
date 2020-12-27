import tkinter as tk
from tkinter import Frame

from UI.R.widgets import selectFramesList


class Viewer:
    def __init__(self):
        self.initWindow()

    def run(self):
        self.win.mainloop()

    def initWindow(self):
        self.win = tk.Tk()
        self.win.title("材料腐蚀预测")  # 添加标题
        self.win.geometry("{}x{}+{}+{}".format(960, 640, 100, 0))
        mainFrame = Frame(self.win, bg="red")

        sFL = selectFramesList(mainFrame, labelTextList=["  模型中心", "训练模型", "校验模型", "预测结果", "  数据中心", "数据集管理"])
        sFL.pack()
        mainFrame.pack()


if __name__ == '__main__':
    v = Viewer()
    v.run()
