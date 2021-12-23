"""
字符串S和T， 在S中找出T所有字母的最小子串

1、我们在字符串 S 中使用双指针中的左右指针技巧，初始化 left = right = 0，把索引闭区间 [left, right] 称为一个「窗口」。
2、我们先不断地增加 right 指针扩大窗口 [left, right]，直到窗口中的字符串 符合要求(包含了 T 中的所有字符)。
3、此时，我们停止增加 right，转而不断增加 left 指针缩小窗口 [left, right]，直到窗口中的字符串不再符合要求(不包含 T 中的所有字符了)。 同时，每次增加 left，我们都要更新一轮结果。
4、重复第 2 和第 3 步，直到 right 到达字符串 S 的尽头。
"""

class Solution:
    def func(self, s, t):
        hashmapNeed = {}
        for c in t:
            if c not in hashmapNeed:
                hashmapNeed[c] = 0
                hashmapNeed[c] += 1
        hashmapWindow = {}
        right = 0
        left = 0
        minLen = float('inf')
        match = 0
        start = 0

        while right < len(s):
            if s[right] in hashmapNeed:
                if s[right] not in hashmapWindow:
                    hashmapWindow[s[right]] = 0
                hashmapWindow[s[right]] += 1
                if hashmapWindow[s[right]] == hashmapNeed[s[right]]:
                    match += 1
            right += 1


            while match == len(t):
                if right - left < minLen:
                    start = left
                    minLen = right - left
                if s[left] in hashmapNeed:
                    hashmapWindow[s[left]] -= 1
                    if hashmapWindow[s[left]] < hashmapNeed[s[left]]:
                        match -= 1
                left += 1


        if minLen == float('inf'):
            return ""
        else:

            return [s[i] for i in range(start,start+minLen)]

testA = Solution()
print(testA.func("abcde", "bcd"))

