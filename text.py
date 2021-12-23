# from pack.mod1 import mod # mod is func in mod1
# from pack import mod1 # mod1 is a .py file in pack
from collections import deque
from pack.packson import *
from pack import *
import sys
'''
gj
'''
print(dir())


queue = deque(["Eric", "John", "Michael"])
queue.popleft()
print(queue)
print(sys.path)
print(dir())

class increaseNum:
    def __iter__(self):
        self.a = 1
        return self
    def __next__(self):
        if self.a < 20:
            x = self.a
            self.a += 1
            return x
        else:
            raise StopIteration

IncreaseNum = increaseNum()
myiter = iter(IncreaseNum)

for x in myiter:
    print(x)





