class Solution:
    def func(self,nums):
        if len(nums) == 1:
            return max(0, nums[0])
        dp = nums[0]
        maxSum = nums[0]
        for i in range(1, len(nums)):
            dp = max(dp, 0) + nums[i]
            maxSum = max(dp, maxSum)
        return maxSum

a = Solution()
print (a.func([-1, 2, 3, -3, 4]))
"""
class Solution:
    def func(self, nums):
        # 1. 状态定义
        dp = [0]* len(nums)
        # 2. 状态初始化
        dp[0] = nums[0]
        ans = nums[0]
        # 3.状态转移 dp[i] = max(dp[i-1],0)+nums(i) dp[i]指1 i+1位 的最大子序和
        for i in range(1, len(nums)):
            dp[i] = max(dp[i-1],0)+nums[i]
            ans = max(dp[i], ans)
        return ans
    # dp[i]只是一个状态值，所以可以不做成数组

    def func2(self, nums):
        # 1. 状态定义
        # 2. 状态初始化
        countSum = nums[0]
        ans = nums[0]
        # 3.状态转移 dp[i] = max(dp[i-1],0)+nums(i) dp[i]指1 i+1位 的最大子序和
        for i in range(1, len(nums)):
            countSum = max(countSum, 0) + nums[i]
            ans = max(countSum, ans)
        return ans
"""





