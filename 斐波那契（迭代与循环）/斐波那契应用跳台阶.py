# 一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。

# dp[0] = 0 dp[1] = 1 dp[2]=2
# dp[n] = dp[n-1]+dp[n-2]


class Solution:
    def func(self, n):
        if n == 0:
            return 0
        elif n == 1:
            return 1
        cur1 = 1
        cur2 = 1
        for _ in range(n-2):
            substitute = cur1
            cur1 = cur2
            cur2 = cur1 + substitute
        return cur2



