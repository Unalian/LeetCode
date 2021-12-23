class Solution:
    def func(self, nums, target):
        hashmap = dict()
        for i, num in enumerate(nums):
            if target - num in hashmap:
                return [hashmap[target - num], i]
            else:
                hashmap[num] = i
        return
testA = Solution()

print(testA.func([2, 7, 11, 15], 9))