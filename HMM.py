class Node(object):
    """ Inits a Node object 
    Pre-conditions: 
    title: string
    observations: observational (string * float) list """
    def __init__(self, title, observations):
        self.title = title
        oDict = {}
        for o, p in observations:
            oDict[o] = p
        self.oDict = oDict
    
    """ Defines equality operator of nodes. Nodes are equal if they have the 
    same title.  
    Pre-conditions:
    [other]: an object of type Node. 
    """
    def __eq__(self, other):
        return self.title == other.title
    
    """ Returns the likelihood of the given [observation]
    Pre-conditions:
    observation: is an observation recorded by the node. """
    def b(observation):
        return self.oDict[observation]


class Graph(object):
    """ Initializes a Graph object
    Pre-conditions:
    nodes: list of nodes, the first node being the starting node (must be root),
           last one being the final node
    children: 2D matrix of nodes * floats, indexed by parent node. Must 
              have same length as [nodes]"""
    def __init__(self, nodes, children):
        self.graphDict = {}
        self.graphList = nodes
        self.root = nodes[0]
        self.end = nodes[-1]
        for i in range(len(nodes)):
            self.graphDict[nodes[i]] = children[i]

    # --- Setters & Getters ---
    """ Gets root of the graph """
    def getRoot():
        return graph.root

    """ Gets end node of the graph """
    def getEnd():
        return graph.end
    
    """ Sets new root of the graph 
    Pre-conditions:
    newRoot: is an orphan node that exists already in the graph """
    def setRoot(newRoot):
        graph.root = newRoot

    """ Sets new end node of the graph 
    Pre-conditions:
    newEnd: is an orphan node that exists already in the graph """
    def setEnd(newEnd):
        graph.end = newEnd

    """ Returns the size of the graph """
    def size():
        return len(self.graphDict)

    """ Returns the list of nodes of the graph"""
    def list():
        return self.graphList
    
    # --- Real methods ---
    """ Adds nodes to the graph and connections to its children. 
    Pre-conditions:
    nodes: list of nodes that contains no node already in the graph
    children: 2D matrix of nodes * floats, indexed by parent node. Must 
              have same length as [nodes] """
    def addNodes(nodes, children):
        self.graphList = self.graphList + nodes
        for i in range(len(nodes)):
            self.graphDict[nodes[i]] = children[i]
    
    """ Removes nodes from the graph and connections to its children.
    Pre-conditions:
    nodes: list of node objects which are nodes that exist in the graph
    children: 2D matrix of nodes * floats, indexed by parent node. Must 
              have same length as [nodes]
    returns void """
    def delNodes(nodes):
        for n in nodes:
            del self.graphDict[n]
            try: 
                self.graphList.remove(n)
            except: 
                pass
        
    """ Returns the weight between nodes [parent] and [child]. 
    Pre-conditions: 
    parent: is a node in the graph. 
    child: is a node in the graph. There must be a direct connection from 
           [parent] to [child]. """
    def a(parent, child):
        children = self.graphDict[parent]
        for x, y in children:
            if x == child:
                return y
        return 0 # if x not in children
    

"""Executes Forward Algorithm on graph object graph
Pre-conditions:
obList: set of all possible observations from every node of [graph]
graph: graph of N states representing a L-R HMM (each path has a weight
       representing a probability & all paths travel in one direction.
returns the probability of observing the observation sequence [obList] in [graph]"""
def FwdAlg(obList, graph):
    # Initialize
    fwd = [[None] * (graph.size() + 2)] * len(obList)
    for i in range(graph.size()):
        s = graph.list()[i]
        fwd[1][i] = graph.a(0, s) * s.b(obList[i])
    # Iteration
    for dt in range(len(obList)-2):
        t = dt + 1
        for i in range(graph.size()):
            s = graph.list()[i]
            evalStep = lambda x : fwd[t-1][x] * graph.a(x, s) * s.b(obList[t]) 
            fwd[t][s] = sum(map(evalStep, fwd[t-1]))
    # Termination TODO: fix for iteration (copy bwd)
    termStep = lambda s:fwd[len(obList)][s]*graph.a(s,graph.getEnd())
    termStep = lambda s : fwd[len(objList)-1][s] * graph.a[graph.list()[s], graph.getEnd()]
    return sum(map(termStep, range(len(graph.list()))))


""" Executes Viterbi Algorithm on a HMM graph object
Pre-conditions:
obList: set of all possible observations from every node of [graph]
graph: graph of N states representing a L-R HMM (each path has a weight
       representing a probability & all paths travel in one direction.
returns the path most likely to observe observation sequence [obList] in [graph]"""
def ViterbiAlg(obList, graph):
    # Initialize matrices
    vit = [[None] * (graph.size() + 2)] * len(obList)
    backpoint = [[None] * (graph.size() + 2)] * len(obList)
    for i in range(graph.size()):
        s = graph.list()[i]
        vit[1][i] = graph.a(0, s) * s.b(obList[i])
        backpoint[1][i] = 0
    # Iteration/Calculation
    for t in range(len(obList)-2):
        for i in range(graph.size()):
            s = graph.list()[i]
            evalStep = lambda x : vit[t-1][x] * graph.a(x, s) * s.b(obList[t]) 
            vit[t][s] = max(map(evalStep, vit))
            for j in range(graph.size()):
                if evalStep j == vit[t][s]:
                    backpoint[t][s] = j
                    break
    # Termination
    vit[-1][i] = 
        max(map(lambda s:vit[len(obList)-1][s]*graph.a(s,graph.list()[-1]), vit))
    backpoint[-1][i] = 
        for j in range(graph.size()):
            if evalStep j = vit[-1][i]:
                backpoint[-1][i] = j
                break
    x = backpoint[-1][i]
    # BackTrace calculation from backpoint matrix
    vitBackTrace = []
    t = -1
    while x != 0:
        vitBackTrace.prepend(x)
        x = backpoint[x][t-1]
        t = t - 1
    return vitBackTrace.prepend(0)


""" Executes Backward Algorithm on graph object [graph] given observation seq
[obList] and returns a ß value. ß representing the probability of seeing the 
observation sequence from time t+1 to end time T given the current state @ time
t and given the current graph. 
Pre-conditions:
[obList]: if length of obList is d, function returns ß[T-d-1:T]
          sequence of vocab of set of all possible observations from [graph]
[graph]: graph of N states representing an L-R HMM

returns probability of observing the observation sequence [obList] in [graph]"""
def BwdAlg(obList, graph):
    # Initialize matrices
    # A/weight matrix from [t..T]
    BMatrix = [[None] * (graph.size() + 2)] * len(obList)
    finalRow = []
    qf = graph.getEnd()
    for node in graph.list():
        finalRow.append(graph.a(node, qf))
    BMatrix[-1] = finalRow
    # Iteration
    for dt in range(len(obList))-1:
        t = len(obList) - dt - 1
        for i in range(graph.size()):
            s = graph.list()[i]
            evalStep = lambda j : graph.a(j,s) * s.b(obList[t+1]) * BMatrix[t+1][j]
            fwd[t][s] = sum(map(evalStep, fwd[t+1]))
    # Termination
    termStep = lambda j : graph.a(0, j) * graph.list()[j].b(obList[0]) * BMatrix[0][j]
    return sum(map(termStep, range(len(graph.list()))))


#TODO: Implement Fwd-Bwd Algo 