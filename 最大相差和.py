class Solution:
    def func(self, prices):
        if prices == []:
            return 0

        maxprofile = 0
        lowest = max(prices)
        for i in prices:
            if i < lowest:
                lowest = i
            else:
                maxprofile = max(i - lowest, maxprofile)
        return maxprofile


a = Solution()
print (Solution.func(a, [2,1,4,1,0,1,2]))