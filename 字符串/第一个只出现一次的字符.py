# 求字符串中第一个只出现一次的字符

class Solution:
    def func(self, string):
        dirc1 = {} # 第一次出现
        dirc2 = {} # 出现次数

        for i in range(len(string)):
            CH = string[i]
            if not dirc1.get(CH):
                dirc1[CH] = i
            if CH not in dirc2:
                dirc2[CH] = 1
            else:
                dirc2[CH] += 1
        frontSeq = float('inf')
        frontC =""
        for C, times in dirc1.items():
            if times == 1:
                if dirc1[C] < frontSeq:
                    frontC = C
                    frontSeq = dirc1[C]
        return frontC

class Solution2:
    def func2(self, string):
        count = {}
        firstLoc = {}
        for i, CH in enumerate(string):
            firstLoc[CH] = firstLoc[CH] if firstLoc.get(CH) else i
            count[CH] = count[CH]+1 if count.get(CH) else 1
        frontSeq = float("inf")
        for CH in count.keys():
            if count.get(CH) == 1 and frontSeq > firstLoc[CH]:
                frontSeq = firstLoc[CH]
        return string[frontSeq]


a = Solution2()
print(a.func2("abcdbcadv"))








