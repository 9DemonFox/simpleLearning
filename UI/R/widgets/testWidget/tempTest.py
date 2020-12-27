from tkinter import *

from tkinter import Frame


class Viewer:
    def __init__(self):
        self.initWindow()

    def run(self):
        self.win.mainloop()

    def initWindow(self):
        self.win = Tk()
        self.win.title("材料腐蚀预测")  # 添加标题
        self.win.geometry("{}x{}+{}+{}".format(960, 640, 100, 0))
        mainFrame = Frame(self.win, bg="red")
        canvas = Canvas(self.win, width=200, height=180, scrollregion=(0, 0, 520, 520))  # 创建canvas
        canvas.place(x=75, y=265)  # 放置canvas的位置
        frame = Frame(canvas)  # 把frame放在canvas里
        frame.place(width=180, height=180)  # frame的长宽，和canvas差不多的
        vbar = Scrollbar(canvas, orient=VERTICAL)  # 竖直滚动条
        vbar.place(x=180, width=20, height=180)
        vbar.configure(command=canvas.yview)
        hbar = Scrollbar(canvas, orient=HORIZONTAL)  # 水平滚动条
        hbar.place(x=0, y=165, width=180, height=20)
        hbar.configure(command=canvas.xview)
        canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)  # 设置
        canvas.create_window((90, 240), window=frame)  # create_window
        canvas.pack()


if __name__ == '__main__':
    v = Viewer()
    v.run()
