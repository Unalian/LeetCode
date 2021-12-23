class TreeNode:
    def __init__(self, item=None, left=None, right=None):
        self.item = item
        self.left = left
        self.right = right


B = TreeNode('B')
C = TreeNode('C')
A = TreeNode('A', B, C)


def tra_rec(root):
    if root is None:
        return
    print(root.item, end=' ')
    tra_rec(root.left)
    tra_rec(root.right)


def tra_norec(root):
    if root is None:
        print("tree is clear")
        return
    list_Node = [root]
    while len(list_Node) != 0:
        node = list_Node.pop()
        print(node.item)
        if node.right is not None:
            list_Node.append(node.right)
        if node.left is not None:
            list_Node.append(node.left)


tra_norec(A)

