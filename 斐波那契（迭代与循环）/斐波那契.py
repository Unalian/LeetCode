class Solution:
    def Fibonacci(self, n):
        queue = [0, 1, 1]
        for i in range(3, n):
            queue.append(queue[i-1]+queue[i-2])
        return queue[n-1]

class Solution2:
    def Fibonacci2(self, n):
        if n == 0:
            return 0
        elif n == 1:
            return 1
        cur1 = 0
        cur2 = 1
        for i in range(n-2):
            substitute = cur1
            cur1 = cur2
            cur2 = substitute + cur1
        return cur2

a = Solution2()
print(a.Fibonacci2(9))