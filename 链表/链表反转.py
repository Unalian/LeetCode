# 反转链表，如果是整个反转，函数式意思是反转以head为开头的链表。递归式为reverse(head.next)
# 基础情况是head = None 或者到达最后一个节点，return head本身就行了
# 栈底情况是，head 和 反转好的head.next为首链表，新的链表头是new_head = reverse(head.next)
# 所以要进行的操作head.next.next = head；head.next = None，把head也加入新链表头引领的新链表

# 其实可以把这个情况想成后序遍历，每次操作就是将一个被遍历的节点反转


def reverse(head):
    if head.next is None or head is None:
        return head
    else:
        new_head = reverse(head.next)
        head.next.next = head
        head.next = None
    return new_head


RECORD = None


# 反转前n个，函数的含义是 反转head为头节点的前n个节点组成的链表。
# 栈底情况是new_head = reverse(head.next, n-1)
# 栈顶情况是n==1时，这就是新的头节点 同时，因为最初传入的head不再指向None。而是指向new_head .next，所以要把这个位置记录下来，在栈底情况中赋予


# 后序遍历理解方法：同样是后序遍历，每个节点翻转， 但是不一样的地方有两点，第一，基础情况由n判断，递归式reverse(head.next, n-1)，n==1时，取得new.head。
# 第二，最初的head节点，它的next不再是None，而是new_head.next。所以在基础情况中，需要记录new_head.next为recode
# 最后，最初的head指向recode就行了。


def reverse_front_n(head, n):
    global RECORD
    if n == 1:# 基础情况
        RECORD = head.next
        return head
    else:
        new_head = reverse_front_n(head.next, n - 1) # 递归推导式子
        head.next.next = head
        head.next = RECORD
    return new_head


# 递归传入 head.next, m-1, n-1, 当m==1，重新变成已经解决的recerse(head, n)
# 返回已将反转好的new_head（后面也已经链接好了），head.next -> new_head
def reverse_m_n(head, m ,n):
    if m == 1:
        return reverse_front_n(head, n)
    else:
        head.next = reverse_m_n(head.next, m-1 ,n-1)
        return head



class Node:
    def __init__(self, item):
        self.next = None
        self.item = item


A = Node(1)
B = Node(2)
C = Node(3)
D = Node(4)
A.next = B
B.next = C
C.next = D


def show_chain(head):
    p = head
    while p is not None:
        print(p.item)
        p = p.next


show_chain(reverse_m_n(A,2,3))
