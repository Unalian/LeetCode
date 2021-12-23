
class Node:
    """define the node"""

    def __init__(self, item):
        # save item
        self.item = item
        # next
        self.next = None


class SingleLinkList:
    """single"""

    def __init__(self):
        self._head = None

    def build_by_list(self,list):
        if len(list) == 0:
            return
        self._head = Node(list[0])
        note_tail = self._head
        for i in range(1, len(list)):
            new_node = Node(list[i])
            note_tail.next = new_node
            note_tail = new_node

    def is_empty(self):
        """if empty"""
        return self._head is None

    def length(self):
        """get length"""
        # head
        cur = self._head
        count = 0
        # if none, get the tail
        while cur is not None:
            count += 1
            #
            cur = cur.next
        return count

    def items(self):
        """go through(iter)"""

        cur = self._head
        # circle
        while cur is not None:
            # return one item
            yield cur.item
            # move
            cur = cur.next


    def add(self, item):
        """add to the head"""
        node = Node(item)
        node.next = self._head
        self._head = node

    def append(self, item):
        """add at the front"""
        node = Node(item)
        # if empty?
        if self.is_empty():
            # if empty: node is the head
            self._head = node
        else:
            # find the tail
            cur = self._head
            while cur.next is not None:
                cur = cur.next
            cur.next = node

    def insert(self, index, item):
        """insert"""
        # if position is behind the first, insert at first
        if index <= 0:
            self.add(item)
        # if index out, insert at the tail
        elif index > (self.length() - 1):
            self.append(item)
        else:
            # creat the node
            node = Node(item)
            cur = self._head
            # to the certain position
            for i in range(index - 1):
                cur = cur.next
            node.next = cur.next
            cur.next = node

    def left_pop(self):
        """left pop"""
        cur = self._head
        if cur is None or cur.next is None:
            return None
        self._head = cur.next
        cur.next = None

    def pop(self):
        """pop"""
        cur = self._head
        if cur is None or cur.next is None:
            return None

        while cur.next.next is not None:
            cur = cur.next
        cur.next = None

    def remove(self, item):
        """delete by item"""
        cur = self._head
        pre = None
        while cur is not None:
            # find the node to delete
            if cur.item == item:
                # hit it( the first)
                if not pre:
                    # the head is pointed at the second node
                    self._head = cur.next
                else:
                    # use pre to delete
                    pre.next = cur.next
                return True
            else:
                # move
                pre = cur
                cur = cur.next


    def find(self, item):
        """if exit"""
        return item in self.items()

    def reverse_list(self):
        """reverse : without recursion"""
        if self._head is None:
            return _head
        cur = _head
        pre = None
        nxt = cur.next
        while nxt:
            cur.next = pre
            pre = cur
            cur = nxt
            nxt = nxt.next
        cur.next = pre
        return cur

    def tail(self):
        """find the last one node"""
        cur = self._head
        while cur.next is not None:
            cur = cur.next
        return cur

    def recursiveReverse(self):
        """reverse"""
        cur = self._head
        def reverse_rev(cur):
            if cur is None or cur.next is None:
                return cur
            else:
                new_head = reverse_rev(cur.next)
                cur.next.next = cur
                cur.next = None
                return new_head
        self._head = reverse_rev(cur)




"""if __name__ == '__main__':
    link_list = SingleLinkList()
    # 向链表尾部添加数据
    for i in range(5):
        link_list.append(i)
    # 向头部添加数据
    link_list.add(6)
    # 遍历链表数据
    for i in link_list.items():
        print(i, end='\t')
    # 链表数据插入数据
    link_list.insert(3, 9)
    print('\n', list(link_list.items()))
    # 删除链表数据
    link_list.remove(0)
    # 查找链表数据
    print(link_list.find(4))
    # pop:
    link_list.pop
    # left pop:
    link_list.left pop"""






