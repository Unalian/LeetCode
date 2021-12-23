

class Solution:
    def lengthOfLongestSubstring(self, s):
        list_record = set()
        l, r, res = -1, -1, 0
        while l < len(s)-1:
            l += 1
            if l != 0:
                list_record.remove(s[l-1])

            while r+1 < len(s) and s[r + 1] not in list_record:
                r += 1
                list_record.add(s[r])
            res = max(res, r-l+1)
        return res

test = Solution()
print(test.lengthOfLongestSubstring("pwwkew"))


