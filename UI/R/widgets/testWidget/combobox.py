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

        sFL = selectFramesList(mainFrame, lambda x : print(x))
        sFL.pack()
        mainFrame.pack()


if __name__ == '__main__':
    v = Viewer()
    v.run()