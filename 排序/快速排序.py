class Solution:
    def participite(self, listSort, left, right):
        girl = left+1
        standard = listSort[left]
        for boy in range(left+1, right+1):
            if listSort[boy] <= standard:
                listSort[boy], listSort[girl] = listSort[girl], listSort[boy]
                girl += 1

        listSort[left], listSort[girl-1] = listSort[girl-1], listSort[left]
        return girl-1

    def quikSort(self, listSort, left, right):
        if left >= right:
            return
        pi = self.participite(listSort, left, right)

        self.quikSort(listSort, left, pi-1)
        self.quikSort(listSort, pi+1, right)

        return listSort




test = Solution()
listSort = [4, 2, 3, 5, 8, 1, 9, 0, -3]
print(test.quikSort(listSort, 0, 8))

