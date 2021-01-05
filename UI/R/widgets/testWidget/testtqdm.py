from time import sleep

from tqdm import tqdm

bar = tqdm(["a", "b", "c", "d"])
for i in bar:
    sleep(0.1)
    bar.set_description("正在打印 %s" % i)
    bar.write("asd")

progress = tqdm(range(5*3 ) )
for i in range(5):
    print('=============================')
    for j in range(3):
        progress.update()
        progress.refresh()
progress.close()