# -*- encoding = utf-8 -*-


class Node:

    def __init__(self, e):
        self.e = e
        self.left = None
        self.right = None


class BSTree:

    def __init__(self):
        self.size = 0
        self.root = None

    def _add(self, node, e):
        if node is None:
            self.size += 1
            node = Node(e)
            return node
        if e < node.e:
            node.left = self._add(node.left, e)
        elif e > node.e:
            node.right = self._add(node.right, e)
        return node

    def add(self, e):
        self.root = self._add(self.root, e)

    def contains(self, node, e):
        if node is Node:
            return False

        if e < node.e:
            return self.contains(node.left, e)
        elif e > node.e:
            return self.contains(node.right, e)
        elif e == node.e:
            return True

    def _minimum(self, node):
        if node.left is None:
            return node
        return self._minimum(node.left)

    def _maximum(self, node):
        if node.right is None:
            return node
        return self._maximum(node.right)

    def removeMin(self, node):
        if node.left is None:
            right = node.right
            node.right = None
            del node
            self.size -= 1
            return right
        node.left = self.removeMin(node.left)
        return node

    def removeMax(self, node):
        if node.right is None:
            left = node.left
            node.left = None
            del node
            self.size -= 1
            return left
        node.right = self.removeMax(node.right)
        return node

    def _remove(self, node, e):
        if node is None:
            return None
        if e < node.e:
            node.left = self._remove(node.left, e)
            return node
        elif e > node.e:
            node.right = self._remove(node.right, e)
            return node
        else:
            if node.left is None:
                Right = node.right
                node.right = None
                del node
                self.size -= 1
                return Right

            if node.right is None:
                Left = node.left
                node.left = None
                del node
                self.size -= 1
                return Left
            successor = self._minimum(node.right)
            successor.right = self.removeMin(node.right)
            successor.left = node.left

            node.left = None
            node.right = None

            return successor

    def remove(self, e):
        self.root = self._remove(self.root, e)

    def min(self):
        return self._minimum(self.root).e

    def max(self):
        return self._maximum(self.root).e

    def delMax(self):
        self.removeMax(self.root)

    def delMin(self):
        self.removeMin(self.root)


if __name__ == '__main__':
    btree = BSTree()
    li = [10 , 6, 12, 5, 7, 11, 14, 1, 7]
    for i in li:
        btree.add(i)
    btree.remove(12)
    print(btree.max())