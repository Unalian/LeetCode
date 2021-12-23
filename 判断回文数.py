

class Solution:
    def func(self, x):

        if x < 0 or (x != 0 and x % 10 == 0):
            return False
        reves = 0
        while x > reves: # reseve是后半部分，x是前半部分
            reves = x%10 + 10* reves
            x = int(x / 10)
        reves2 = int(reves / 10)# 当位数为奇数， 除掉多余当一位
        return (reves == x) or  reves2 == x

a = Solution()
print(a.func(123321))


"""pay attention to the int type"""