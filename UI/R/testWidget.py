import tkinter as tk
from tkinter import Frame

from UI.R.widgets import selectFramesList,baseSelectFrameListEvent

class SelectFrameListEvent(baseSelectFrameListEvent):
    def __init__(self):
        super().__init__()
        pass

    def recall(self, text):
        print(text)


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

        handler = SelectFrameListEvent()
        sFL = selectFramesList(mainFrame, lambda x : print(x))
        sFL.pack()
        mainFrame.pack()


if __name__ == '__main__':
    v = Viewer()
    v.run()
