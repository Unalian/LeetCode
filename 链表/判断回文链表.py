class Node:
    def __init__(self,item):
        self.item = item
        self.next = None

A = Node(1)
B = Node(2)
C = Node(2)
D = Node(2)
A.next = B
B.next = C
C.next = D


### 判断回文链表：其实就是后序遍历链表right，然后和全局变量left比较，每次right往前一次，left就往后一次
LEFT = None
def is_back_equal(head):
    global LEFT
    LEFT = head
    print(traverse(head))



def traverse(head):
    global LEFT
    if(head == None):
        return True
    RES = traverse(head.next)
    RES = RES and (head.item == LEFT.item)
    LEFT = LEFT.next
    return RES




is_back_equal(A)


