class Solution:
    def findSub3(self,nums):
        output0 = [[]]
        output = [[]]
        for num in nums:
            output0 += [[num] + _ for _ in output0]
        for i in output0:
            i.sort()
            if i not in output:
                output.append(i)
        return output

a = Solution()
print (Solution.findSub3(a, [1,2,2]))