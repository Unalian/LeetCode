# 给定一棵二叉树，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。
# 广度优先搜索：在递归的操作部分迭代right[depth]的值
from collections import deque

class TreeNode:
    def __init__(self, item=None, left=None, right=None):
        self.item = item
        self.left = left
        self.right = right


def build_tree(inorder, postorder):
    if not postorder:
        return
    root = TreeNode(postorder[-1])
    rootid = inorder.index(root.item)

    root.left = build_tree(inorder[:rootid], postorder[:rootid])
    root.right = build_tree(inorder[rootid + 1:], postorder[rootid:-1])
    return root


def traverse(root):
    if root is None:
        return
    print(root.item)
    traverse(root.left)
    traverse(root.right)

A= build_tree(['d','b','a','c','e'], ['d', 'b', 'e', 'c', 'a'] )

class Solution:
    def rightSideView(self, root: TreeNode):
        que = deque([(root, 0)])
        depth_dict = dict()
        while len(que) != 0:
            node, depth = que.popleft()
            if node is not None:
                depth_dict[depth] = node.item
                que.append((node.left, depth+1))
                que.append((node.right, depth+1))
        for depth in depth_dict.keys():
            print(depth, depth_dict[depth])


test = Solution()
test.rightSideView(A)

