from collections import deque


class TreeNode:
    def __init__(self, item = None, left = None, right = None):
        self.item = item
        self.left = left
        self.right = right


B = TreeNode('B')
C = TreeNode('C')
A = TreeNode('A', B, C)


def tra(root):
    if root is None:
        return
    tra(root.left)
    print(root.item)
    tra(root.right)

tra(A)

"""
def tra_bfs(root):
    if root is None:
        print("clear tree")
        return
    que = deque([root])
    while len(que) != 0:
        node = que.popleft()
        print(node.item)
        if node.left is not None:
            que.append(node.left)
        if node.right is not None:
            que.append(node.right)


tra_bfs(A)
"""