
# 先做前n个反转的递归，再递归入上一层k个一组

class Node:
    def __init__(self, value = None, next_node = None):
        self.value = value
        self.next = next_node


class Solution:
    next_node = None

    def res_n(self, head, n):
        if n == 1:
            self.next_node = head.next
            return head
        new_head = self.res_n(head.next, n-1)
        head.next.next = head
        head.next = self.next_node
        return new_head


    def can_res(self, head, k):
        if head is None:
            return False
        for i in range(k-1):
            if head.next is None:
                return False
        return True


    def res_k(self, head, k):
        if self.can_res(head, k) is False:
            return head
        new_head = self.res_n(head, k)
        head.next = self.res_k(self.next_node, k)
        return new_head

def traverse(head):
    if head is None:
        print("empty chain")
        return
    print(head.value)
    traverse(head.next)



A = Node(1)
B = Node(2)
C = Node(3)
D = Node(4)
E = Node(5)
F = Node(6)
G = Node(5)
H = Node(6)

A.next = B
B.next = C
C.next = D
D.next = E
E.next = F
F.next = G
G.next = H

test = Solution()
traverse(test.res_k(A, 2))