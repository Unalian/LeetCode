# 给定一个数组A[0,1,…,n-1],请构建一个数组B[0,1,…,n-1],其中B中的元素B[i]=A[0]A[1]…*A[i-1]A[i+1]…*A[n-1]。不能使用除法。

class Solution:
    def func(self, listA):
        listB = [1]
        n = len(listA)
        for i in range(1, n):
            listB.append(listB[i-1]*listA[i-1])
        cur = 1
        for j in range(n-2, -1, -1):
            cur *= listA[j+1]
            listB[j] *= cur
        return listB

a = Solution()
print(a.func([1, 2]))