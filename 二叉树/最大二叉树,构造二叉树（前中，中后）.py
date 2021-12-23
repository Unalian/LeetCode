class TreeNode:
    def __init__(self, item=None, left=None, right=None):
        self.item = item
        self.left = left
        self.right = right


## 最大二叉树
# 二叉树的根是数组中的最大元素。
# 左子树是通过数组中最大值左边部分构造出的最大二叉树。
# 右子树是通过数组中最大值右边部分构造出的最大二叉树。


def max_tree(nums):
    if len(nums)  == 0:
        return None
    maxdata = -float('inf')
    maxdataIndex = -1
    for i, item in enumerate(nums):
        if item > maxdata:
            maxdata = item
            maxdataIndex = i
    root = TreeNode(maxdata)
    root.left = max_tree(nums[:maxdataIndex])
    root.right = max_tree(nums[maxdataIndex + 1:])
    return root



def traverse(root):
    if root is None:
        return
    print(root.item)
    traverse(root.left)
    traverse(root.right)


# 中后
def buildTree(inorder, postorder):
    if not postorder:
        return
    root = TreeNode(postorder[-1])
    rootid = inorder.index(root.item)

    root.left = buildTree(inorder[:rootid], postorder[:rootid])
    root.right =buildTree(inorder[rootid+1:], postorder[rootid:-1])
    return root


# 前中
def reConstructBinaryTree(pre, tin):#pre、tin分别是前序序列和中序序列
    # write code here
    if len(pre)>0:
        root = TreeNode(pre[0])#前序序列的第一个肯定是当前子树的根节点
        rootid = tin.index(root.item)#通过根节点在中序序列中的位置划分出左右子树包含的节点
        root.left = reConstructBinaryTree(pre[1:1+rootid], tin[:rootid])#重建左子树
        root.right = reConstructBinaryTree(pre[1+rootid:], tin[rootid+1:])#重建右子树
        return root

traverse(buildTree(['d','b','a','c','e'], ['d', 'b', 'e', 'c', 'a'] ))

