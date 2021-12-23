"""
设计一个支持 push ，pop ，top 操作，并能在常数时间内检索到最小元素的栈。

push(x) —— 将元素 x 推入栈中。
pop() —— 删除栈顶的元素。
top() —— 获取栈顶元素。
getMin() —— 检索栈中的最小元素。
"""


class Solution:
    def __init__(self):
        self.stack = []
        self.min_record = [float("inf")]

    def push(self, x):
        record = x if x < self.min_record[-1] else self.min_record[-1]
        self.stack.append(x)
        self.min_record.append(record)

    def pop(self):
        self.stack.pop()
        self.min_record.pop()

    def top(self):
        return self.stack[-1]

    def getMin(self):
        return self.min_record[-1]


def main():
    test = Solution()
    test.push(1)
    test.push(3)
    test.push(5)
    test.pop()
    print(test.getMin())
    print(test.top())

main()

