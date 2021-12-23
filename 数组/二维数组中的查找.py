# 在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，
# 输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。


#
class Solution:
    def func(self, n, matrix):
        if matrix is False:
            return False
        row = len(matrix)
        col = len(matrix[0])
        i = row-1
        j = 0
        while i != 0 or j != col:
            if matrix[i][j] > n:
                i = i-1
                if i < 0:
                    return False
            elif matrix[i][j] < n:
                j = j+1
                if j > col:
                    return False
            else:
                return True
        if matrix[i][j] == n:
            return True
        else:
            return False

class Solution2:
    def func2(self, n, matrix):
        if matrix is False:
            return False
        row, col = len(matrix), len(matrix[0])
        i, j = row-1, 0
        while i >= 0 and j <= col-1:
            if matrix[i][j] == n:
                return True
            elif matrix[i][j] > n:
                i -= 1
            else:
                j += 1
        return False


a = Solution2()
print(a.func2(2, [[1, 2, 4], [2, 4, 6]]))




