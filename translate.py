
class Solution:

    def translate(self,oldlen):
        dirNum = {'1':'一', '2':'二','3':'三','4':'四','5':'五','6':'六','7':'七','8':'八','9':'九','0':"零"}
        k = 0
        newlen = [" "]*len(oldlen)
        for i in oldlen:
            i = str(i)
            if len(i) == 1:
                newlen[k] = dirNum[i]
            else:
                if i[0] == "1":
                    if i[1] == '0':
                        newlen[k] = "十"
                    else:
                        newlen[k] ="十" + dirNum[i[1]]
                elif i[1] == "0":
                    newlen[k] = dirNum[i[0]] + "十"
                else:
                    newlen[k] = dirNum[i[0]] + "十" + dirNum[i[1]]
            k = k+1
        string = " ".join(newlen)+" "
        return string


a = Solution()
n = list(input().split(','))
print(Solution.translate(a, n))
