class TreeNode:
    def __init__(self, item=None, left=None, right=None):
        self.item = item
        self.left = left
        self.right = right


class TreeNodeLink:
    def __init__(self, item=None, left=None, right=None, next = None):
        self.item = item
        self.left = left
        self.right = right
        self.next = next


B = TreeNode('B')
C = TreeNode('C')
A = TreeNode('A', B, C)


# 节点数量


def count_node(root):
    if root is None:
        return 0
    return count_node(root.left) + count_node(root.right) + 1

# 反转


def reverse(root):
    if root is None:
        return None

    root.left, root.right = root.right, root.left


    reverse(root.left)
    reverse(root.right)
    return root

# 将每一层的节点链接起来


def link_node(root: TreeNodeLink):
    if root is None or root.left is None:
        return root
    link_two_node(root.left, root.right)
    return root


def link_two_node(node1: 'TreeNodeLink', node2: 'TreeNodeLink'):
    if node1 is None:
        return
    node1.next = node2
    link_two_node(node1.left, node1.right)
    link_two_node(node2.left, node2.right)
    link_two_node(node1.right, node2.left)

#  将二叉树展开为链表
# 以下流程：
# 1、将 root 的左子树和右子树拉平。
# 2、将 root 的右子树接到左子树下方，然后将整个左子树作为右子树。


def tree_to_chain(root):
    if root is None:
        return
    tree_to_chain(root.left)
    tree_to_chain(root.right)

    left = root.left
    right = root.right
    root.left = None
    root.right = left
    p = root
    while p.right is not None:
        p = p.right
    p.right = right






link_node(A)
tree_to_chain(A)
print(count_node(reverse(A)))


