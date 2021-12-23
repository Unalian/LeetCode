class Solution:
    def func(self, x):
        res = 0
        k = 0
        intMax = pow(2,31) -1
        while x!= 0:
            if x < 0:
                k = 1
                x = -x
            pop = x%10
            x = int(x/10)
            if (res > intMax/10) or (res == intMax/10 and pop> 7) : # 保证不溢出
                return 0
            res = res*10 + pop
        if k == 1:
            res = -res
        return res

testA = Solution()
print(testA.func(-12345))
