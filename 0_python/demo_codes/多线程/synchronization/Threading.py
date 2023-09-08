#!/usr/bin/python3

import threading
import time

exitFlag = 0

class myThread(threading.Thread):
    def __init__(self,threadID,counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = threading.Thread.getName(self)
        self.counter = counter

    def run(self):
        print("start Thread:"+self.name)
        threadLock.acquire() 				##使用在前，定义在后
        print_time(self.name, self.counter,5)
        print("stop Thread:"+self.name)
        threadLock.release()		

def print_time(threadName,delay,counter):
    while counter:
        if exitFlag:
            threadName.exit()
        time.sleep(delay)
        print("%s: %s" % (threadName, time.ctime(time.time())))
        counter -= 1

threadLock = threading.Lock() ##这里就是全局变量了，没有必要定义在前
threads = []

thread1 = myThread(1,2)
thread2 = myThread(2,4)

threads.append(thread1)
threads.append(thread2)

for t in threads:
	t.start()

for t in threads:
	t.join()

print("退出主线程!")
        
