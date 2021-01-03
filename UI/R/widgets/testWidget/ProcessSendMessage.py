#!coding:utf-8
import random
import time
from multiprocessing import Process, Queue
from tqdm import tqdm

# 写数据进程执行的代码
def proc_write(q, urls):
    print('Process is write....')
    for url in tqdm(urls):
        q.put(url)
        print('put {} to queue... '.format(url))
        time.sleep(random.random())


# 读数据进程的代码
def proc_read(q):
    print('Process is reading...')
    while True:
        url = q.get(True)
        print('Get %s from queue' % url)


if __name__ == '__main__':
    # 父进程创建Queue，并传给各个子进程
    q = Queue()
    proc_write1 = Process(target=proc_write, args=(q, ['url_1', 'url_2', 'url_3']))
    proc_write2 = Process(target=proc_write, args=(q, ['url_4', 'url_5', 'url_6']))
    proc_reader = Process(target=proc_read, args=(q,))
    # 启动子进程，写入
    proc_write1.start()
    proc_write2.start()
    proc_reader.start()
    # 等待proc_write1结束
    proc_write1.join()
    proc_write2.join()
    # proc_raader进程是死循环，强制结束
    proc_reader.terminate()
