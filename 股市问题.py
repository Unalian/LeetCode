

class Socket:

    # 可以购买一次
    def func1(self, prices):
        dp=[[0 for _ in range(2)]for _ in prices]
        dp[0][0] = 0
        dp[0][1] = -prices[0]
        for i in range(1,len(prices)):
            dp[i][0] = max(dp[i-1][0],dp[i-1][1]+prices[i])
            dp[i][1] = max(dp[i-1][1],-prices[i])
        return dp[len(prices)-1][0]
    def func12(self,prices):
        dp_i_0 = 0
        dp_i_1 = -prices[0]
        for i in range(1,len(prices)):
            dp_i_0 = max(dp_i_0, dp_i_1+prices[i])
            dp_i_1 = max(dp_i_1,-prices[i])
        return dp_i_0

    # 可以购买无数次
    def func1000(self,prices):
        dp_i_0 = 0
        dp_i_1 = -prices[0]
        for i in range(1,len(prices)):
            temp = dp_i_0
            dp_i_0 = max(dp_i_0, dp_i_1 + prices[i])
            dp_i_1 = max(dp_i_1, temp-prices[i])
        return dp_i_0

    # 可以购无数次但是要隔一天买
    def funcSpace(self, prices):
        dp_i_0 = 0
        dp_i_1 = -100
        dp_i_pre = 0
        for i in range(1, len(prices)):
            temp = dp_i_0
            dp_i_0 = max(dp_i_0, dp_i_1 + prices[i])
            dp_i_1 = max(dp_i_1, dp_i_pre - prices[i])
            dp_i_pre = dp_i_0


    # 可以购无数次但是要收税
    def func1000withtax(self,prices):
        dp_i_0 = 0
        tax = 1
        dp_i_1 = -prices[0]
        for i in range(1,len(prices)):
            temp = dp_i_0
            dp_i_0 = max(dp_i_0, dp_i_1 + prices[i]-tax)
            dp_i_1 = max(dp_i_1, temp-prices[i])
        return dp_i_0

    # 可以购买两次 可以穷举
    def func2(self,prices):
        max_k = 2
        dp=[[[0 for _ in range(2)]for _ in range(0,max_k+1)]for _ in prices]


        for i in range(0, len(prices)):
            for k in range(1,max_k+1):
                if i-1 == -1:
                    dp[i][k][0] = 0
                    dp[i][k][1] = -prices[0]
                    # dp[i][k][0] = Math.max(dp[i - 1][k][0], dp[i - 1][k][1] + prices[i]);
                    # dp[i][k][1] = Math.max(dp[i - 1][k][1], dp[i - 1][k - 1][0] - prices[i]);
                    continue
                dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])
                dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0]-prices[i])
        print(dp)
        return dp[len(prices)-1][max_k][0]


a = Socket()
print(Socket.func2(a,[12,14,13,21,20,30]))






