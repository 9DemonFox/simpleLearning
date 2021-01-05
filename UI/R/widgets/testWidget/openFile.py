#!/usr/bin/python
# -*-coding:utf-8 -*-

from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
import tkinter as tk
from _ast import If

top = tk.Tk()
# 这里四个参数分别为：宽、高、左、上
top.geometry("500x300+750+200")

top.title("www.tianqiweiqi.com")

strPath = StringVar()
strResult = StringVar()


def pathCallBack():
    filePath = filedialog.askopenfilename();
    if (filePath != ''):
        strPath.set(filePath);


def okCallBack():
    strResult = 'i love you!'
    txtResult.delete(0.0, tk.END)
    txtResult.insert(tk.INSERT, strResult)
    txtResult.update()


btnPath = tk.Button(top,
                    text='选择',
                    width=10,
                    command=pathCallBack)
btnOk = tk.Button(top,
                  text='转换',
                  width=10,
                  command=okCallBack)
Label(top, text="图片路径：").grid(row=0, column=0)
Entry(top, width=45, textvariable=strPath).grid(row=0, column=1)
btnPath.grid(row=0, column=2);

Label(top, text="文本内容：").grid(row=2, column=0)
txtResult = Text(top, width=45, height=15)

txtResult.grid(row=3, column=1)
txtResult.insert(tk.END, 'Do you love me?')

btnOk.grid(row=4, column=2);

top.mainloop();