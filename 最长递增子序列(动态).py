# 给定一个无序的整数数组，找到其中最长上升子序列的长度。
# 动态规划： dp[n] 指以n结尾的子序列的最长递增序列长度
# 转移方程是 dp[n] = dp[j] + 1(j 遍历0，n-1 如果nums[n]>nums[j] and dp[j] is max)
# max(dp) 可以求数组最大值
"""子序列-一维数组"""
class Solution:

    def the_max_raise_son_que(self, nums):
        dp = []
        dp.append(1)
        for n in range(1, len(nums)):
            dp.append(1)
            for j in range(n-1, -1, -1):
                if nums[n] > nums[j] and dp[n] < dp[j] + 1:
                    dp[n] = dp[j] + 1
        return max(dp)

a = Solution()
print(a.the_max_raise_son_que([0,1,0,3,2,3]))




