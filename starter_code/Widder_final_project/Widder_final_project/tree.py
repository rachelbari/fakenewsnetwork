import random
from itertools import chain

UNK = 'UNK'
PARSER_TAGS = [
    "S:", "SBAR:", "SBARQ:", "SINV:", "SQ:", "ADJP:", "ADVP:", "CONJP:",
    "FRAG:", "INTJ:", "LST:", "NAC:", "NP:", "NX:", "PP:", "PRN:", "PRT:",
    "QP:", "RRC:", "CP:", "VP:", "WHADJP:", "WHAVP:", "WHADVP:", "WHNP:",
    "WHPP:", "CC:", "CD:", "DT:", "EX:", "FW:", "IN:", "JJ:", "JJR:", "JJS:",
    "LS:", "MD:", "NN:", "NNS:", "NNP:", "NNPS:", "PDT:", "POS:", "PRP:",
    "PRP$:", "RB:", "RBR:", "RBS:", "RP:", "SYM:", "TO:", "UH:", "VB:", "VBD:",
    "VBG:", "VBN:", "VBP:", "VBZ:", "WDT:", "WP:", "WP$:", "WRB:", "X:", "#:",
    "$:", "\":", "(:", "):", ",:", "::", "``:", ".:", "'':", "UCP:", "-RRB-:",
    "-LRB-:"
]

# This file contains the dataset in a useful way. We populate a list of
# Trees to train/test our Neural Nets such that each Tree contains any
# number of Node objects.

# The best way to get a feel for how these objects are used in the program is to drop pdb.set_trace() in a few places throughout the codebase
# to see how the trees are used.. look where loadtrees() is called etc..


class Node:  # a node in the tree
    def __init__(self, label, word=None):
        self.label = label
        self.word = word
        self.parent = None  # reference to parent
        # self.left = None  # reference to left child
        # self.right = None  # reference to right child
        self.children = []
        # true if I am a leaf (could have probably derived this from if I have
        # a word)
        self.isLeaf = False
        # true if we have finished performing fowardprop on this node (note,
        # there are many ways to implement the recursion.. some might not
        # require this flag)
        self.probs = None
        self.h = None


class Tree:
    def __init__(self, line_string, openChar='(', closeChar=')'):
        tokens = []
        self.open = '('
        self.close = ')'
        # parse line
        filename, tree = line_string.split(' x--x ')
        # get vote label from filename
        vote = filename[-5]
        vote_num = 0 if vote == "N" else 4

        for toks in tree.strip().split():
            if toks == "ROOT:":
                tokens += [vote_num]
            elif toks in PARSER_TAGS:
                tokens += [2]
            else:
                tokens += [toks]

        self.root = self.parse(tokens)[0]
        # get list of labels as obtained through a post-order traversal
        self.labels = get_labels(self.root)

    def parse(self, tokens, parent=None):
        if not tokens:
            return []

        # clean up mistakes in parsing
        if tokens[0] != '(':
            return self.parse(tokens[1:], parent=parent)
        if tokens[-1] != ')':
            tokens = tokens + [')']
        if len(tokens) < 4:
            return []

        split = 2  # position after open and label
        countOpen = countClose = 0

        if tokens[split] == self.open:
            countOpen += 1
            split += 1

        # Find where first child and remaining children split
        while countOpen != countClose and len(tokens) > split:
            if tokens[split] == self.open:
                countOpen += 1
            if tokens[split] == self.close:
                countClose += 1
            split += 1

        # New node
        if type(tokens[1]) == int:
            node = Node(tokens[1])  # zero index labels
        else:
            print("Error!")
            print(tokens)
            return self.parse(tokens[1:], parent=parent)

        node.parent = parent

        # leaf Node
        if countOpen == 0:
            node.word = tokens[2]
            node.isLeaf = True
            if len(tokens) > 4:
                return [node] + self.parse(tokens[4:], parent=parent)
            return [node]

        node.children = self.parse(tokens[2:split], parent=node)
        if len(tokens) > split and tokens[split] == self.close:
            return [node] + self.parse(tokens[split + 1:], parent=parent)
        node.children += self.parse(tokens[split:-1], parent=node)

        return [node]

    def get_words(self):
        leaves = getLeaves(self.root)
        words = [node.word for node in leaves]
        return words


def leftTraverse(node, nodeFn=None, args=None):
    """
    Recursive function traverses tree
    from left to right.
    Calls nodeFn at each node
    """
    if node is None:
        return
    for child in node.childen:
        leftTraverse(child, nodeFn, args)
    nodeFn(node, args)


def getLeaves(node):
    if node is None:
        return []
    if node.isLeaf:
        return [node]
    else:
        return [item for child in node.children for item in getLeaves(child)]


def get_labels(node):
    labels = []
    if node is None:
        return labels
    if node.isLeaf:
        labels = [node.label]
    else:
        labels += [node.label]
        for child in node.children:
            labels += get_labels(child)
    return labels


def clearFprop(node, words):
    node.fprop = False


def loadTrees(dataSet='train'):
    """
    Loads training trees. Maps leaf node words to word ids.
    """
    path = './project_data'
    file = "{}/parsed_{}.txt".format(path, dataSet)

    print("Loading %s trees.." % dataSet)
    with open(file, 'r') as fid:
        trees = [Tree(l) for l in fid.readlines()]
    return trees


def simplified_data(num_train, num_dev, num_test):
    rndstate = random.getstate()
    random.seed(0)

    # TO USE IDEOLOGICALLY BIASED TREES, UNCOMMENT BELOW
    # trees = loadTrees('train') + loadTrees('dev') + loadTrees('test')

    # TO USE EXTREME (SUPPORT/OPPOSE) TREES, UNCOMMENT BELOW
    # trees = loadTrees('extreme')

    # TO USE TREES FROM ONE BILL, UNCOMMENT ONE LINE FROM BELOW
    trees = loadTrees('006')
    # trees = loadTrees('013')
    # trees = loadTrees('016')

    #filter root labels
    yes_trees = [t for t in trees if t.root.label == 4]
    no_trees = [t for t in trees if t.root.label == 0]

    #split into train, dev, test
    print(len(yes_trees), len(no_trees))
    random.shuffle(yes_trees)
    random.shuffle(no_trees)
    yes_trees = sorted(yes_trees, key=lambda t: len(t.get_words()))
    no_trees = sorted(no_trees, key=lambda t: len(t.get_words()))
    num_train /= 2
    num_dev /= 2
    num_test /= 2
    train = yes_trees[:int(num_train)] + no_trees[:int(num_train)]
    dev = yes_trees[int(num_train):int(num_train) + int(
        num_dev)] + no_trees[int(num_train):int(num_train) + int(num_dev)]
    test = yes_trees[int(num_train) + int(num_dev):int(num_train) + int(
        num_dev) + int(num_test)] + no_trees[int(num_train) + int(num_dev):int(
            num_train) + int(num_dev) + int(num_test)]
    random.shuffle(train)
    random.shuffle(dev)
    random.shuffle(test)
    random.setstate(rndstate)

    return train, dev, test
