class Graph(object):
    """ Initializes a Graph object
    Example: ('HOT1', ('HOT1',))
    Pre-conditions:
    nodes: list of nodes, the first node being the starting node (must be root),
           last one being the final node
    children: 2D matrix of nodes * floats, indexed by parent node. Must 
              have same length as [nodes]
    observations: indexed sequence of observation * prob lists corresponding to 
                  node index"""
    def __init__(self, data):
        # field declarations
        self.graphDict = {}
        self.Q = [x[0] for x in data]
        self.A = [[0 for x in range(len(data))] for x in range(len(data))]
        self.V = set()
        self.B = []
        # field instantiation
        for i in range(len(data)):
            _, children, obs = data[i]
            for child, prob in children:
                try: 
                    j = self.Q.index(child)
                    self.A[i][j] = prob
                except ValueError:
                    print("%s (child) isn't found in the node list" %child)
            self.V = self.V | set([e[0] for e in obs])
            self.B.append(obs)
        self.start = data[0][0]
        self.final = data[-1][0]

# --- Setters & Getters ---
    """ Gets root of the graph """
    def getStart(self):
        return self.start

    """ Gets end node of the graph """
    def getFinal(self):
        return self.final
    
    """ Sets new root of the graph 
    Pre-conditions:
    newStart: is an orphan node that exists already in the graph """
    def setStart(self, newStart):
        self.start = newStart

    """ Sets new end node of the graph 
    Pre-conditions:
    newFinal: is an orphan node that exists already in the graph """
    def setFinal(self, newFinal):
        self.final = newFinal

    """ Returns the size of the graph """
    def size(self):
        return len(self.Q)

    """ Returns the list of nodes of the graph"""
    def list(self):
        return self.Q
    
    """ Returns the [i]th element of the graph
    Pre-conditions
    [i]: i < self.size()"""
    def elt(self, i):
        return self.Q[i]
    
    """ Gets A of the graph """
    def getA(self):
        return self.A

    """ Gets B of the graph """
    def getB(self):
        return self.B
    
    """ Sets new A of the graph 
        Pre-conditions:
        newA: is an orphan node that exists already in the graph """
    def setA(self, newA):
        self.A = newA

    """ Sets new B of the graph 
        Pre-conditions:
        newB: is an orphan node that exists already in the graph """
    def setB(self, newB):
        self.B = newB
    
    """ Returns the index of [node] in the graph, returns -1 if [node] doesn't
        exist in the graph
        Pre-conditions: [node] is a string """
    def index(self, node):
        try:
            return self.Q.index(node)
        except ValueError:
            print("%s was not found in graph's Q set" %node)
            return -1
        
    
# --- Real methods ---
    """ Returns the weight between nodes [parent] and [child]. 
        Pre-conditions: 
        parent: is a node in the graph. 
        child: is a node in the graph. There must be a direct connection from 
            [parent] to [child]. """
    def a(self, parent, child, ints=False):
        if ints:
            i, j = parent, child
        else: 
            try:
                i = self.Q.index(parent)
                j = self.Q.index(child)
            except ValueError:
                print("%s or %s were not found in graph's Q set" %(parent, child))
                raise ValueError()
        return self.A[i][j]

    """ Returns the likelihood of the given [observation]
    Pre-conditions:
    state: is a node in the graph
    observation: is an observation recorded by the node. """
    def b(self, state, observation, ints=False):
        if ints:
            i = state
        else:
            try:
                i = self.Q.index(state)
            except ValueError:
                print("%s was not found in the graph" %state)
                raise ValueError()
        obList = self.B[i]
        for k, v in obList:
            if observation == k:
                return v
        return 0


"""Executes Forward Algorithm on graph object graph
    Pre-conditions:
    obList: set of all possible observations from every node of [graph]
    graph: graph of N states representing a L-R HMM (each path has a weight
        representing a probability & all paths travel in one direction.
    returns the probability of observing the observation sequence [obList] in [graph]
    alpha(t, q) where q is implicitly the final state, and t is the length of [obList]"""
def FwdAlg(obList, graph, final=None, T=None):
    final = final if final != None else graph.getFinal()
    T = T if T != None else len(obList)
    # Initialize
    fwd = [[0.0 for z in range(graph.size())] for z in range(len(obList))]
    for i in range(graph.size()):
        s = graph.elt(i)
        fwd[0][i] = graph.a(graph.getStart(), s) * graph.b(s,obList[0])
    # Iteration
    for dt in range(T-1):
        t = dt + 1
        for i in range(graph.size()):
            s = graph.elt(i)
            evalStep = lambda x : fwd[t-1][x] * graph.a(graph.elt(x), s) * graph.b(s, obList[t])
            fwd[t][i] = sum(map(evalStep, range(graph.size())))
    # Termination
    termStep = lambda s : fwd[T-1][s] * graph.a(graph.elt(s), graph.getFinal())
    fwd[T-1][-1] = sum(map(termStep, range(graph.size())))
    print(fwd)
    return fwd[T-1][graph.index(final)]


""" Executes Viterbi Algorithm on a HMM graph object
    Pre-conditions:
    obList: set of all possible observations from every node of [graph]
    graph: graph of N states representing a L-R HMM (each path has a weight
        representing a probability & all paths travel in one direction.
    returns the path most likely to observe observation sequence [obList] in [graph]"""
def ViterbiAlg(obList, graph):
    # Initialize matrices
    vit = [[None for x in graph.size()] for x in range(len(obList))]
    backpoint = [[None for x in graph.size()] for x in range(len(obList))]
    for i in range(graph.size()):
        s = graph.elt(i)
        vit[1][i] = graph.a(0, s) * s.b(obList[i])
        backpoint[1][i] = 0
    # Iteration/Calculation
    for t in range(len(obList)-2):
        for i in range(graph.size()):
            s = graph.elt(i)
            evalStep = lambda x : vit[t-1][x] * graph.a(x, s) * s.b(obList[t]) 
            vit[t][s] = max(map(evalStep, vit))
            for j in range(graph.size()):
                if evalStep(j) == vit[t][s]:
                    backpoint[t][s] = j
                    break
    # Termination
    termStep = lambda s:vit[len(obList)-1][s]*graph.a(graph.elt(s),graph.getFinal())
    vit[-1][-1] = max(map(termStep, range(graph.size())))
    for j in range(graph.size()):
        if termStep(j) == vit[-1][-1]:
            backpoint[-1][-1] = j
            break
    x = backpoint[-1][-1]
    # BackTrace calculation from backpoint matrix
    vitBackTrace = []
    t = -1
    while x != 0:
        vitBackTrace.prepend(x)
        x = backpoint[x][t-1]
        t = t - 1
    return vitBackTrace.prepend(0)


""" Executes Backward Algorithm on graph object [graph] given observation seq
    [obList] and returns a Beta value. Beta representing the probability of seeing the 
    observation sequence from time t+1 to end time T given the current state @ time
    t and given the current graph. 
    Pre-conditions:
    [obList]: if length of obList is d, function returns B[T-d-1:T]
            sequence of vocab of set of all possible observations from [graph]
    [graph]: graph of N states representing an L-R HMM
    returns probability of observing the observation sequence [obList] in [graph]"""
def BwdAlg(obList, graph):
    # Initialize matrices
    BMatrix = [[0.0 for z in range(graph.size())] for z in range(len(obList))]
    finalRow = []
    qf = graph.getFinal()
    for node in graph.list():
        finalRow.append(graph.a(node, qf))
    BMatrix[-1] = finalRow
    # Iteration
    for dt in range(len(obList)-1):
        t = len(obList) - dt - 2
        for i in range(graph.size()):
            s = graph.elt(i)
            evalStep = lambda j : graph.a(graph.elt(j),s) * graph.b(s, obList[t+1]) * BMatrix[t+1][j]
            print(t, i, s)
            BMatrix[t][i] = sum(map(evalStep, range(graph.size())))
    # Termination
    termStep = lambda j : graph.a(graph.getStart(), graph.elt(j)) * graph.b(graph.elt(j),obList[0]) * BMatrix[1][j]
    print(BMatrix)
    return sum(map(termStep, range(graph.size())))


""" Executes the xi function
"""
def xi(t, i, j, obList, graph):
    fwdObs = obList[:t]
    bwdObs = obList[t+1:]
    alph = FwdAlg(fwdObs, graph)
    beta = BwdAlg(bwdObs, graph)
    a = graph.a(i, j, True)
    b = graph.b(j, t, True)
    fullalpha = FwdAlg(obList, graph)
    return (alph * a * b * beta)/fullalpha


# #TODO: Implement Fwd-Bwd Algo 
# def forward_backward(obList, state set = )