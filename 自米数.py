class Solution:
    def is_flower(self, num):
        num_string = str(num)
        length = len(num_string)
        num_sum = 0
        for i in num_string:
            num_sum += int(i) ** length
        if num_sum == num:
            return True
        else:
            return False

    def find_flowers(self, max_numbers):
        result = []
        for i in range(1, max_numbers+1):
            if self.is_flower(i) is True:
                result.append(i)
        return result

# 样例测试

import sys
def is_flower(num):
    num_string = str(num)
    length = len(num_string)
    num_sum = 0
    for i in num_string:
        num_sum += int(i) ** length
    if num_sum == num:
        return True
    else:
        return False


def find_flowers(min_numbers, max_numbers):
    result = []
    for i in range(min_numbers, max_numbers + 1):
        if is_flower(i) is True:
            result.append(i)
    if len(result) == 0:
        return ["no"]
    return result


def read_data():
    try:
        data = []
        while True:
            one_line = list(map(int, sys.stdin.readline().split()))
            if len(one_line) == 0:
                break
            data.append(one_line)
        return data
    except:
        pass

def main():
    data = read_data()
    for data_min_max in data:
        for i in find_flowers(data_min_max[0], data_min_max[1]):
            print(i, end=" ")
        print("")


main()

