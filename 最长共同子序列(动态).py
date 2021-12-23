# 题目：
# 给定两个字符串 text1 和 text2，返回这两个字符串的最长公共子序列的长度。
#
# 一个字符串的 子序列 是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。
# 例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。两个字符串的「公共子序列」是这两个字符串所共同拥有的子序列。
#
# 若这两个字符串没有公共子序列，则返回 0。
"""
思路：
对于两个字符串的动态规划问题，一般来说都是像本文一样定义 DP table， 因为这样定义有一个好处，就是容易写出状态转移方程， dp[i][j] 的状态 可以通过之前的状态推导出来:
定义二阶数组：dp含义：dp[i][j] s1中 第一个到第i个数字组成的数组 和 s2中 第一个数字到第j个数字组成的数组（序号0--j-1）的数组的最长序列长度。
决定是 s1[i-1] == s2[j-1]:
状态转移方程是 dp[i][j] = dp[i-1][j-1]+1
初始化是 dp[0][...] = dp[...][0] = 0
"""
"""子序列-二维数组"""


import sys
class Solution:
    def func(self, text1, text2):
        dp = [[0] * (len(text2) + 1) for _ in range(len(text1) + 1)]
        for s1_seq in range(1, len(text1) + 1):
            for s2_seq in range(1, len(text2) + 1):
                if text1[s1_seq - 1] == text2[s2_seq - 1]:
                    dp[s1_seq][s2_seq] = dp[s1_seq-1][s2_seq-1] + 1
                else:
                    dp[s1_seq][s2_seq] = max(dp[s1_seq-1][s2_seq], dp[s1_seq][s2_seq-1])
        return dp[len(text1)][len(text2)]

a = Solution()
print(a.func([1,2,3,4,7],[2,3,4,5,7]))