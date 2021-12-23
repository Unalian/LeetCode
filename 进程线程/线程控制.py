 #!/usr/bin/env python3
# coding:utf-8
from atexit import register
from random import randrange
from threading import Thread, current_thread, Lock
from time import sleep, ctime


class CleanOutputSet(set):
    def __str__(self):    #改变输出格式
        return ', '.join(x for x in self)


lock = Lock()
pause_time = (2, 3, 4, 5)
remaining = CleanOutputSet()


def loop(nsec):
    myname = current_thread().name    #获取当前线程名
    lock.acquire()
    remaining.add(myname)    #添加到集合
    print('[{0}] Start {1}'.format(ctime(), myname))
    lock.release()
    sleep(nsec)
    lock.acquire()
    remaining.remove(myname)
    print('[{0}] Completed {1} ({2} secs)'.format(ctime(), myname, nsec))
    print('  (remaining: {0})'.format(remaining or 'None'))
    lock.release()

def main():
    for pause in pause_time:
        Thread(target=loop, args=(pause,)).start()


@register
def _atexit():
    print('all DONE at:', ctime())


if __name__ == '__main__':
    main()
