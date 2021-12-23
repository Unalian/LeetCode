"""
动态-子串-二维
"""
class Solution:
    def func(self, s):
        ans = ""
        dp = [[False for _ in range(len(s))] for _ in range(len(s))]
        for i in range(len(s)-1,-1,-1):
            for j in range(i,len(s)):
                if i == j:
                    dp[i][j] = True
                elif i+1 == j:
                    dp[i][j] = (s[i] == s[j])
                else:
                    dp[i][j] = dp[i+1][j-1] and (s[i] == s[j])
                if j-i+1 > len(ans) and dp[i][j]:
                    ans = s[i:j+1]
        return ans





testA  = Solution()
print(testA.func("abaabaaabala"))