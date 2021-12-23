
"""
import sys
class Solution:










listOfPara = list(map(int, (sys.stdin.readline().strip()).split()))
n = listOfPara[0]
m = listOfPara[1]
c = listOfPara[2]
listOfColor = []
for _ in range(listOfPara[0]):
    listOfColor.append(list(map(int, (sys.stdin.readline().strip()).split())))
print(listOfPara)
print(listOfColor)

a = Solution()
a.func(n, m, c, tuple(listOfColor))
"""