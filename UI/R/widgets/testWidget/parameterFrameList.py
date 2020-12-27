import tkinter as tk
from tkinter import Frame

from UI import R
from UI.Modeler import Modeler


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
        parameterList = [('惩罚项系数 alpha', 0.001), ('参数淘汰率 k', 0.25)]
        m = Modeler()
        sFL = R.widgets.ParameterFrameList(mainFrame, parameterList)
        sFL.pack()
        # sFL.bind(lambda event, c: print(c.getAllParameters()), sFL)
        sFL.bind(lambda event, c: c.repack([(' alpha', 0.001), (' k', 0.25), ("asd", "123")]), sFL)
        mainFrame.pack()


if __name__ == '__main__':
    v = Viewer()
    v.run()
