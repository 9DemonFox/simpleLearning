import tkinter as tk
from tkinter import Frame

from UI.R.widgets import ParameterFrame
from UI import Modeler

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
        sFL = ParameterFrame(mainFrame, "参数淘汰率", "0.25")
        sFL.pack()
        mainFrame.pack()


if __name__ == '__main__':
    v = Viewer()
    v.run()
