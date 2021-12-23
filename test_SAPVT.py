import sys
import math


def func():

    input_list = read_data()
    for input_m_n in input_list:
        n = input_m_n[0]
        res = n
        for _ in range(input_m_n[1]-1):
            n = math.sqrt(n)
            res += n
        print("%.2f" % res)


def read_data():
    try:
        mx = []
        while True:
            m = list(map(int, sys.stdin.readline().strip().split()))
            # m = sys.stdin.readline().strip()
            # 若是多输入，strip()默认是以空格分隔，返回一个包含多个字符串的list。
            if len(m) == 0:
                break
            mx.append(m)
        return mx
    except:
        pass

func()