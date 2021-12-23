class Node:
    def __init__(self, key=-1, value=-1, pre=None, nex=None):
        self.key = key
        self.value = value
        self.pre = pre
        self.next = nex


class LRU:
    dirt_key_value = {}
    chain_head = Node()
    chain_last = Node(-1, -1, chain_head)
    chain_head.next = chain_last
    capacity_used = 0

    def __init__(self, capacity = None):
        self.capacity = capacity

    def get(self, key):
        if key in self.dirt_key_value.keys():
            self.set_mid_node_last(self.dirt_key_value[key])
            return self.dirt_key_value[key].value
        else:
            print("No such key-value")
            return None

    def set(self, key, value):
        if key in self.dirt_key_value.keys():
            self.dirt_key_value[key].value = value
            self.set_mid_node_last(self.dirt_key_value[key])
        else:
            if self.capacity_used == self.capacity:
                self.chain_action_remove_first()
                self.capacity_used -= 1
            self.capacity_used += 1
            new_node = Node(key, value)
            self.dirt_key_value[key] = new_node
            self.chain_action_put_last(new_node)


    def chain_action_put_last(self, node_put_last):
        node_put_last.next = self.chain_last
        node_put_last.pre = self.chain_last.pre
        self.chain_last.pre.next = node_put_last
        self.chain_last.pre = node_put_last

    def chain_action_remove_mid(self, node_mid):
        node_mid.pre.next = node_mid.next
        node_mid.next.pre = node_mid.pre

    def chain_action_remove_first(self):
        self.dirt_key_value.pop(self.chain_head.next.key)
        self.chain_head.next = self.chain_head.next.next
        self.chain_head.pre = self.chain_head

    def set_mid_node_last(self, node_mid):
        self.chain_action_remove_mid(node_mid)
        self.chain_action_put_last(node_mid)


def traserve(head):
    if head is None:
        return
    print(head.key)
    traserve(head.next)



test = LRU(2)
test.set(1, 1)
test.set(2, 2)
test.set(3, 3)
traserve(test.chain_head)


