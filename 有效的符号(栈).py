class Solution:
    def func(self, s):
        if len(s) % 2 != 0:
            return False
        hashmap = {'(':')', "[":"]", "{":"}"}
        stack = []
        for f in s:
            if f in hashmap.keys():
                stack.append(f)
            else:
                if len(stack) == 0:
                    return False
                elif hashmap[stack[-1]] == f:
                    stack.pop()
                else:
                    return False
        if len(stack) == 0:
            return True
        else:return False

testA = Solution()
print(testA.func("){"))