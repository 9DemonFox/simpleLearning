import threading
import time
import tkinter
from multiprocessing import Process
from multiprocessing import Queue
from tkinter import ttk


class UpdateValue(threading.Thread):
    def __init__(self, threadname, mq, pb):
        threading.Thread.__init__(self, name=threadname)
        self.threadname = int(threadname)
        self.mq = mq
        self.process_bar = pb

    def run(self):
        while True:
            if not self.mq.empty():
                message = self.mq.get()
                self.process_bar.now = int(message)


class MinProgressBar(Process):

    def __init__(self):
        self.message_queue = Queue(10)
        super().__init__(target=MinProgressBar.msg_consumer, args=(self, self.message_queue))
        self.now = 0
        self.start()

    def msg_consumer(self, mq):
        root = tkinter.Tk()
        root.geometry("{}x{}+{}+{}".format(150, 120, 0, 0))

        progressbarOne = ttk.Progressbar(root)
        progressbarOne.pack(pady=20)
        # 进度值最大值
        progressbarOne['maximum'] = 100
        # 进度值初始值
        progressbarOne['value'] = 0

        thread0 = UpdateValue("0", mq, self)
        thread0.start()
        while (self.now < 100):
            time.sleep(0.2)
            progressbarOne['value'] = self.now
            root.update()
        thread0.join()
        root.destroy()
        root.mainloop()


if __name__ == '__main__':

    cons = MinProgressBar()
    cons.start()
    for i in range(100):
        time.sleep(0.1)
        cons.message_queue.put(i + 1)
    cons.terminate()
    cons.join()
    cons.close()
