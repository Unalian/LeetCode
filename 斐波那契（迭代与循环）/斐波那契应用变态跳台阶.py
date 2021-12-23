# 一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。

# dp[n] = dp[0] + dp[1] + dp[2] + ... + dp[n-1] = dp[n-1]*2
class Solution:
    def func(self, n):
        if n == 0:
            return 0
        elif n == 1:
            return 1
        dp = [1, 1]
        for i in range(2, n+1):
            cur = 0
            for j in range(i):
                cur += dp[j]
            dp.append(cur)
        return dp[n]

class Solution2:
    def func2(self, n):
        if n == 0:
            return 0
        if n == 1:
            return 1
        cur = 1
        for i in range(2, n+1):
            cur *= 2
        return cur

class Solution3:
    def func3(self, n):
        if n < 3:
            return n
        else:
            2**(n-1)

a = Solution3()
print(a.func3(4))

