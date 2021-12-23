

class Solution:

    def findSub2(self,nums):

        def back(first = 0, cur = []):
            # 满足条件的操作
            if len(cur) == k:
                output.append(cur)
            for i in range(first,len(nums)) :

                cur.append(nums[i])
                # 递归调整每次参数
                back(i+1, cur)
                # 回退
                cur.pop()
        for k in range(0, n+1):
            back()
        return output


    # 无重复
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


    def findSub1(self, nums):
        output = [[]]
        for num in nums:

            output += [[num] + _ for _ in output]
        return output



a = Solution()
print (Solution.findSub3(a, [1,2,2,3]))





