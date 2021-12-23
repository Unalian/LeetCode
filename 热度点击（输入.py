import sys

class Solution:
    def func(self, n, listOfCus, q, *a):
        for listIn in a:
            for listInn in listIn:
                l = listInn[0]
                r = listInn[1]
                k = listInn[2]
                res = 0
                for i in range(l, r+1):
                    if listOfCus[i-1] == k:
                        res += 1
                print(res)

a = Solution()
n = int(sys.stdin.readline().strip('\n'))
listOfCus = list(map(int,(sys.stdin.readline().strip()).split()))
q = int(sys.stdin.readline().strip('\n'))
listOfQ = []
for _ in range(q):
    listOfQ.append(list(map(int,(sys.stdin.readline().strip()).split())))
tupleOfQ = tuple(listOfQ)

a.func(n, listOfCus, q, tupleOfQ)