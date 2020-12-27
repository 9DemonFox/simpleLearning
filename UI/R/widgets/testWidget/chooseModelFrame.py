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
        mainFrame = Frame(self.win, bg="red")
        sFL = R.widgets.ChooseModelFrame(mainFrame, ["a", "b"])
        sFL.pack()
        sFL.bind(func=lambda event, sFL: print(sFL.getCurModel()), C=sFL)
        mainFrame.pack()


if __name__ == '__main__':
    v = Viewer()
    v.run()
