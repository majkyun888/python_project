#! -*- encoding=utf-8 -*-


class treeNode:

    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}

    def inc(self, numOccur):
        self.count += numOccur

    def disp(self, ind=1):
        print(ind, self.name, '', self.count)
        for child in self.children.values():
            child.disp(ind + 1)


def loadSimpDat():
    simpDat = [
        ['r', 'z', 'h', 'j', 'p'],
        ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
        ['z'],
        ['r', 'x', 'n', 'o', 's'],
        ['y', 'r', 'x', 'z', 'q', 't', 'p'],
        ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']
    ]
    return simpDat


def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict


def updateHeader(nodeToTest, targetNode):
    while nodeToTest.nodeLink != None:
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)
    else:
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)


def createTree(dataSet, minSup=1):
    headrTable = {}
    for trans in dataSet:
        for item in trans:
            headrTable[item] = headrTable.get(item, 0) + dataSet[trans]
    for key in headrTable.keys():
        if headrTable[key] < minSup:
            del headrTable[key]
    freqItemSet = set(headrTable.keys())
    if len(freqItemSet) == 0:
        return None, None
    for k in headrTable:
        headrTable[k] = [headrTable[k], None]
    retTree = treeNode('Null Set', 1, None)  #根树
    for tranSet, count in dataSet.items():
       # print(dataSet.items())
        localD = {}
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headrTable[item][0]
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]     #由大到小
            updateTree(orderedItems, retTree, headrTable, count)
    return retTree, headrTable



simpDat = loadSimpDat()
dataSet = createInitSet(simpDat)
createTree((dataSet))