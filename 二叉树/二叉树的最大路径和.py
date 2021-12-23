class TreeNode:
    def __init__(self, item = None, left = None, right = None):
        self.item = item
        self.left = left
        self.right = right


class Solution:
    maxSum = -float("inf")

    def maxPathSum(self, root):
        self.maxPer(root)
        return self.maxSum

    def maxPer(self, node):
        if node is None:
            return 0
        left = max(self.maxPer(node.left), 0)
        right = max(self.maxPer(node.right), 0)
        self.maxSum = self.maxSum if node.item + left + right < self.maxSum else node.item + left + right
        return node.item + max(left, right)


def traverse(root):
    if root is None:
        return
    print(root.item)
    traverse(root.left)
    traverse(root.right)



test = Solution()
A = TreeNode(1, TreeNode(2), TreeNode(3))
traverse(A)
print(test.maxPathSum(A))
