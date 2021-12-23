import sys
from collections import deque
import heapq
class Solution:
    def func(self,s):
        print(s[1:2+1])
        hashmapNeed = {x:0 for x in s}
        print(hashmapNeed)
        for location, key in enumerate(hashmapNeed):
            print(location)
            print(key)

    def func2 (self):
        # 队列
        dp = [1,2,3]
        que = deque(dp)
        que.popleft()
        dp2 = list(que)
        print(dp2)

    def func3(self):
        list = [1,2,3]
        print(list[:3])


    def func4(self):
        try:
            while True:
                line = sys.stdin.readline().strip()
                if line == '':
                    break
                a = list(map(int, (line).split(' ')))
        except:
            pass


    def func5(self):
        a = sys.stdin.readlines()
        output_thing = []
        for line in a:
            readed_thing = list(map(int, line.split()))
            output_thing.append(readed_thing)
        print(output_thing)


    def fun6(self):
        heap_thing = heapq.heapify([])
        heap_thing.heappush()





# str = list(map(int,sys.stdin.readline().split()))
#print(str)
# print(type(str))

A = Solution()
A.func5()