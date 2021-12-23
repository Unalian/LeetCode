class Solution:
    def func(self, s1, s2, s3):
        dp= [[False for _ in range(len(s2)+1)]for _ in range(len(s1)+1)]
        dp[0][0] = True
        if (len(s1)+len(s2) != len(s3)):
            return False
        for i in range(1, len(s1)+1):
            dp[i][0] = dp[i-1][0] and s3[i-1] == s1[i-1]
        for j in range(1, len(s2)+1):
            dp[0][j] = dp[0][j-1] and s3[j-1] == s2[j-1]
        for i in range(1, len(s1)+1):
            for j in range(1,len(s2)+1):
                dp[i][j] = (dp[i][j-1] and s3[i+j-1] == s2[j-1]) or (dp[i-1][j] and s3[j+i-1]== s1[i-1])
        return dp[len(s1)][len(s2)]

a = Solution()
print(Solution.func(a,"","b","b"))