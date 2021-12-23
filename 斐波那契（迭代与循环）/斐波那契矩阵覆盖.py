# 我们可以用21的小矩形横着或者竖着去覆盖更大的矩形。请问用n个21的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？
# 012 的情况需要单独考虑
# dp[n] = dp[n-1] + dp[n-2]

class Solution:
    def func(self, n):
        if n == 0:
            return 0
        if n == 1:
            return 1
        cur1 = 1
        cur2 = 1
        for _ in range(n-2):
            sub = cur1
            cur1 = cur2
            cur2 = cur1 + sub
        return cur2




