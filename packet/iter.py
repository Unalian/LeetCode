from collections import deque

class iterlity:
    def __next__(self):
        if self.a <= 20:
            x =self.a
            self.a += 1
            return x
    def __iter__(self):
        self.a = 1
        return self

a = iterlity()
b = iterlity().__iter__()
for _ in range(0,3):
    print(b.__next__())
