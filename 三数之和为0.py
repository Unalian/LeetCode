class Solution:
    def func_3sum(self, list_enter, target):
        res = []
        for i in range(len(list_enter)-2):
            res_single = self.func_2sum(list_enter[i+1:], target - list_enter[i])
            for tuple in res_single:
                tuple.append(list_enter[i])
                if tuple not in res:
                    res.append(tuple)

    def func_2sum(self, list_enter, target):
        dirc = {}
        res = []
        for i, num in enumerate(list_enter):
            if (target - num) not in dirc.keys():
                dirc[num] = i
            else:
                res_single = [list_enter[dirc[target-num]], num]
                res_single.sort()
                dirc.pop(target - num)
                if res_single not in res:
                    res.append(res_single)
        return res


a = Solution()
print(a.func_2sum([1, 2, 2, 1, 3, 0, 4, -1], 3))





