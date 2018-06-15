class Node(object):
    """ Inits a Node object 
    Pre-conditions: 
    title: string
    observations: observational (string * float) list """
    def __init__(title, observations):
        self.title = title
        oDict = {}
        for o, p in observations:
            oDict[o] = p
        self.oDict = oDict
    
    """ Returns the likelihood of the given [observation]
    Pre-conditions:
    observation: is an observation recorded by the node. """
    def b(observation):
        return self.oDict[observation]


class Graph(object):
    """ Initializes a Graph object
    Pre-conditions:
    nodes: list of strings
    children: 2D matrix of strings * floats, indexed by parent node. Must 
              have same length as [nodes]"""
    def __init__(nodes, children):
        self.graphDict = {}
        for i in range(len(nodes)):
            self.graphDict[nodes[i]] = children[i]
    
    """ Adds nodes to the graph and connections to its children. If [nodes] 
    contains a string that is already in the graph, it will be overwritten.
    Pre-conditions:
    nodes: list of nodes
    children: 2D matrix of nodes * floats, indexed by parent node. Must 
              have same length as [nodes] """
    def addNodes(nodes, children):
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

    """ Returns the size of the graph """
    def size():
        return len(self.graphDict)

    """ Returns the list of nodes of the graph"""
    def list():
        return list(d.keys())
        
    """ Returns the weight between nodes [initial] and [final]. 
    Pre-conditions: 
    initial: is a node in the graph. 
    final: is a node in the graph. There must be a direct connection from 
           [initial] to [final]. """
    def a(initial, final):
        initialChildren = self.graphDict[initial]
        for x, y in initialChildren:
            if x == final:
                return y
    

"""Executes Forward Algorithm on graph object graph

Pre-conditions:
obList: set of all possible observations from every node of [graph]
graph: graph of N states representing a L-R HMM (each path has a weight
       representing a probability & all paths travel in one direction.
returns the probability of observing the observation sequence [obList] in [graph]"""
def FwdAlg(obList, graph):
    fwd = [[None] * len(obList)] * (graph.size() + 2)
    for i in range(graph.size()):
        s = graph.list()[i]
        fwd[i][1] = graph.a(0, s) * s.b(obList[i])
    for t in range(len(obList)-2):
        for i in range(graph.size()):
            s = graph.list()[i]
            evalStep = lambda x : fwd[x][t-1] * graph.a(x, s) * s.b(obList[t]) 
            fwd[s][t] = sum(map(evalStep, fwd))
    return sum(map(lambda s:fwd[s][len(obList)]*graph.a(s,graph.list()[-1]), fwd))


""" Executes Viterbi Algorithm on a HMM graph object

Pre-conditions:
obList: set of all possible observations from every node of [graph]
graph: graph of N states representing a L-R HMM (each path has a weight
       representing a probability & all paths travel in one direction.
returns the path most likely to observe observation sequence [obList] in [graph]"""
def ViterbiAlg(obList, graph):
    vit = [[None] * len(obList)] * (graph.size() + 2)
    backpoint = [[None] * len(obList)] * (graph.size() + 2)
    for i in range(graph.size()):
        s = graph.list()[i]
        vit[i][1] = graph.a(0, s) * s.b(obList[i])
        backpoint[i][1] = 0
    for t in range(len(obList)-2):
        for i in range(graph.size()):
            s = graph.list()[i]
            evalStep = lambda x : vit[x][t-1] * graph.a(x, s) * s.b(obList[t]) 
            vit[s][t] = max(map(evalStep, vit))
            for j in range(graph.size()):
                if evalStep j == vit[s][t]:
                    backpoint[s][t] = j
                    break
    vit[i][t] = 
        max(map(lambda s:vit[s][len(obList)]*graph.a(s,graph.list()[-1]), vit))
    backpoint[i][t] = 
        for j in range(graph.size()):
            if evalStep j = vit[i][t]:
                backpoint[i][t] = j
                break
    x = backpoint[i][t]
    VitBackTrace = []
    while x != 0:
        VitBackTrace.prepend(x)
        x = backpoint[x][t-1]
    return VitBackTrace.prepend(0)


#TODO: Implement Fwd-Bwd Algo 