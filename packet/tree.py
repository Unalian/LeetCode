class TreeNode(object):
     def __init__(self, data=None,left=None,right=None):
         self.val =data
         self.left =left
         self.right =right

"""
A = TreeNode('A','B','C')
B = TreeNode('B',None,'D')
C = TreeNode('C','E','F')
E = TreeNode('E','G',None)
F = TreeNode('F','H','I')
"""

class tree_binary(object):
    def __init__(self):
        self._head = None

    def levelOrder(self):
        root = self._head
        nodeQuene = []
        result = []
        if root is None:
            return result
        nodeQuene.append(root)
        while nodeQuene:
            singlevel =[] # items for one level
            queneLength = len(nodeQuene)
            for i in range(queneLength):
                currentNode = nodeQuene.pop(0)
                if currentNode.left != None:
                    nodeQuene.append(currentNode.left)
                if currentNode.right != None:
                    nodeQuene.append(currentNode.right)
                singlevel.append(currentNode.val)
            result.append(singlevel)
        return result

    def build_tree_by_list(self, nodeList):
        if nodeList[0] is None:
            return None
        self._head = TreeNode(nodeList[0])
        head = self._head
        Nodes = [head]
        j = 1
        for node in Nodes:
            if node is not None:
                node.left = (TreeNode(nodeList[j]) if nodeList[j] is not None else None)
                Nodes.append(node.left)
                j += 1
                if j == len(nodeList):
                    return head
                node.right = (TreeNode(nodeList[j]) if nodeList[j] is not None else None)
                j += 1
                Nodes.append(node.right)
                if j == len(nodeList):
                    return head

    def buildTree_in_post(self, inorder, postorder):
        def buildTree(inorder, postorder):
            if not postorder:
                return
            for i in range(len(inorder)):
                if inorder[i] == postorder[-1]:
                    root = TreeNode(postorder[-1])
                    root.left = buildTree(inorder[:i], postorder[:i])
                    root.right =buildTree(inorder[i + 1:], postorder[i:-1])
                    return root
        self._head = buildTree(inorder, postorder)

    def preOrderTraverseRec(self):
        cur = self._head
        def preOrder(head):
            if head is None:
                return None
            print(head.val, end = " ")
            preOrder(head.left)
            preOrder(head.right)
        preOrder(cur)

    def preOrderTraverse(self):
        stack = [self._head]
        node = self._head
        while len(stack) > 0:
            print(node.val, end = " ")
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
            node = stack.pop()

    def inOrderTraverseRec(self):
        cur = self._head
        def preOrder(head):
            if head is None:
                return None
            preOrder(head.left)
            print(head.val, end=" ")
            preOrder(head.right)
        preOrder(cur)

    def inOrderTraverse(self):
        stack = []
        pos = self._head
        while pos or len(stack) > 0:
            if pos:
                stack.append(pos)
                pos = pos.left
            else:
                pos = stack.pop()
                print(pos.val, end=" ")
                pos = pos.right

    def postOrderTraverseRec(self):
        cur = self._head
        def preOrder(head):
            if head is None:
                return None
            preOrder(head.left)
            preOrder(head.right)
            print(head.val, end=" ")
        preOrder(cur)

    def postOrderTraverse(self):
        stack = [self._head]
        stack2 = []
        while len(stack) > 0:
            node = stack.pop()
            stack2.append(node)
            if node.left is not None:
                stack.append(node.left)
            if node.right is not None:
                stack.append(node.right)
        while len(stack2) > 0:
            print(stack2.pop().val)

    def layerTraverse(self):
        node = self._head
        if not node:
            return None
        queue = []
        queue.append(node)
        while len(queue) > 0:
            tmp = queue.pop(0)
            print(tmp.val)
            if tmp.left:
                queue.append(tmp.left)
            if tmp.right:
                queue.append(tmp.right)













