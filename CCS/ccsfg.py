"""@package ccsfg
Package @package ccsfg contains the necessary building blocks to implement a bipartite factor graph tailored to belief
propagation. The target application is coded compressed sensing, which often necessitates the use of a large alphabet.
Thus, the structures of @class VariableNode and @class CheckNode assume that messages are passed using either fast
Walsh–Hadamard transform (FWHT) or fast Fourier transform (FFT) techniques.
"""
import numpy as np
# The fast Walsh–Hadamard transform (FWHT) is borrowed from FALCONN, a library for similarity search over
# high-dimensional data.
# https://github.com/FALCONN-LIB/FFHT
# import ffht


class GenericNode:
    """
    Base class @class GenericNode creates a single generic node within a graph.
    This class implements rudimentary properties that are leveraged by derived classes.
    """

    def __init__(self, nodeid, neighbors=None):
        """
        Initialize node of type @class GenericNode.
        :param nodeid: Identifier corresponding to self
        :param neighbors: List of identifiers corresponding to neighbors of self
        """

        # Identifier of self
        self.__id = nodeid
        # List of identifiers corresponding to neighbors within graph
        self.__Neighbors = []
        # Dictionary of messages from neighbors accessed with their identifiers
        # Some neighbors may not have set messages therefore @var self.__Neighbors
        # may not match @var self.__MessagesFromNeighbors.keys()
        self.__MessagesFromNeighbors = dict()

        # Argument @var neighbors (optional), if specified in list form, determines neighbors added
        if neighbors is not None:
            self.addneighbors(neighbors)

    @property
    def id(self):
        return self.__id

    @property
    def neighbors(self):
        return self.__Neighbors

    def addneighbor(self, neighborid, message=None):
        """
        Add neighbor @var neighborid to list of neighbors.
        Add message @var message (optional) to dictionary of messages from neighbors.
        :param neighborid: Identifier of neighbor to be added
        :param message: Message associated with @var neighborid
        """
        if neighborid in self.__Neighbors:
            print('Node ID ' + str(neighborid) + 'is already a neighbor.')
        else:
            if message is None:
                self.__Neighbors.append(neighborid)
            else:
                self.__MessagesFromNeighbors.update({neighborid: message})
                self.__Neighbors.append(neighborid)

    def addneighbors(self, neighborlist):
        """
        Add neighbors whose identifiers are contained in @var neighborlist to list of neighbors.
        :param neighborlist: List of node identifiers to be added as neighbors
        """
        for neighborid in neighborlist:
            self.addneighbor(neighborid)

    def getstate(self, neighborid):
        """
        Output message corresponding to @var nodeid.
        :param neighborid:
        :return:
        """
        if neighborid in self.__MessagesFromNeighbors.keys():
            return self.__MessagesFromNeighbors[neighborid]
        else:
            return None

    def getstates(self):
        """
        Output @var self.__MessagesFromNeighbors in dictionary format.
        :return: Dictionary of messages from neighbors
        """
        return self.__MessagesFromNeighbors

    def setstate(self, neighborid, message):
        """
        set message for neighbor with identifier @var neighborid.
        :param neighborid: Identifier of origin
        :param message: Message corresponding to identifier @var neighborid
        """
        if neighborid in self.__Neighbors:
            self.__MessagesFromNeighbors[neighborid] = message
        else:
            print('Check node ID ' + str(neighborid) + ' is not a neighbor.')


class VariableNode(GenericNode):
    """
    Class @class VariableNode creates a single variable node within bipartite factor graph.
    """

    def __init__(self, varnodeid, messagelength, neighbors=None):
        """
        Initialize variable node of type @class VariableNode.
        :param varnodeid: Unique identifier for variable node
        :param messagelength: Length of incoming and outgoing messages
        :param neighbors: Neighbors of node @var varnodeid in bipartite graph
        """

        super().__init__(varnodeid, neighbors)
        # Length of messages
        self.__MessageLength = messagelength

        # Check node identifier 0 corresponds to trivial check node associated with local observation
        # Initialize messages from (trivial) check node 0 to uninformative measure (all ones)
        self.addneighbor(0, message=np.ones(self.__MessageLength, dtype=float))

    @property
    def neighbors(self):
        # Exclude trivial check node 0 associated with local observation from list of neighbors
        return [neighbor for neighbor in super().neighbors if neighbor != 0]

    def reset(self):
        """
        Reset every state of variable node to uninformative measures (all ones).
        This method employs @property super().neighbors to properly reset message for
        (trivial) check node zero to uninformative measure.
        """
        for neighborid in super().neighbors:
            self.setstate(neighborid, np.ones(self.__MessageLength, dtype=float))
        # self.setobservation(self, np.ones(self.__MessageLength, dtype=float))

    def getobservation(self):
        """
        Retrieve status of local observation (checkneighborid 0)
        :return: Measure of local observation
        """
        return self.getstate(0)

    def setobservation(self, measure):
        """
        Set status of local observation @var self.__CheckNeighbors[0] to @param measure.
        :param measure: Measure of local observation
        """
        self.setstate(0, measure)

    def setmessagefromcheck(self, checkneighborid, message):
        """
        Incoming message from check node neighbor @var checkneighbor to variable node self.
        :param checkneighborid: Check node identifier of origin
        :param message: Incoming belief vector
        """
        self.setstate(checkneighborid, message)

    def getmessagetocheck(self, checkneighborid=None):
        """
        Outgoing message from variable node self to check node @var checkneighborid
        Exclude message corresponding to @var checkneighborid (optional).
        If no destination is specified, return product of all measures.
        :param checkneighborid: Check node identifier of destination
        :return: Outgoing belief vector
        """
        dictionary = self.getstates()
        if checkneighborid is None:
            states = list(dictionary.values())
        elif checkneighborid in dictionary:
            states = [dictionary[key] for key in dictionary if key is not checkneighborid]
        else:
            print('Destination check node ID ' + str(checkneighborid) + ' is not a neighbor.')
            return None

        if np.isscalar(states):
            return states
        else:
            states = np.array(states)
            if states.ndim == 1:
                return states
            elif states.ndim == 2:
                try:
                    return np.prod(states, axis=0)
                except ValueError as e:
                    print(e)
            else:
                raise RuntimeError('Dimenstion: states.ndim = ' + str(np.array(states).ndim) + ' is not allowed.')

    def getestimate(self):
        """
        Retrieve distribution of beliefs associated with self
        :return: Local belief distribution
        """
        measure = self.getmessagetocheck()
        if measure is None:
            return measure
        elif np.isscalar(measure):
            return measure
        else:
            # Normalize only if measure is not zero vector.
            # Numpy function np.isclose() breaks execution.
            weight = np.linalg.norm(measure, ord=1)
            if weight == 0:
                return measure
            else:
                # Under Numpy, division by zero seems to be a warning.
                return measure / np.linalg.norm(measure, ord=1)


class CheckNodeBinary(GenericNode):
    """
    Class @class CheckNodeBinary creates a single check node within a bipartite factor graph.  This
    class is specifically designed for binary LDPC codes in the probability domain. 
    """

    def __init__(self, checknodeid, messagelength, neighbors=None):
        """
        Initialize check node of type @class CheckNodeBinary.
        :param checknodeid: Unique identifier for check node
        :param messagelength: length of messages.  In the binary case, this always equals 2
        :param neighbors: Neighbors of node @var checknodeid in bipartite graph
        """

        super().__init__(checknodeid, neighbors)
        self.__MessageLength = messagelength

    def reset(self):
        """
        Reset check nodes to uninformative measures
        """
        uninformative = np.ones(self.__MessageLength)
        for neighborid in self.neighbors:
            self.setstate(neighborid, uninformative)

    def setmessagefromvar(self, varneighborid, message):
        """
        Incoming message from variable node neighbor @var varneighborid to check node self.
        :param varneighborid: Variable node identifier of origin
        :param message: incoming belief measure
        """

        self.setstate(varneighborid, message)

    def getmessagetovar(self, varneighborid):
        """
        Outgoing message from check node self to variable node @var varneighbor
        :param varneighborid: Variable node identifier of destination
        :return: Outgoing belief measure
        """

        dictionary = self.getstates()
        if varneighborid is None:
            states = list(dictionary.values())
        elif varneighborid in dictionary:
            states = [dictionary[key] for key in dictionary if key is not varneighborid]
        else:
            print('Destination variable node ID ' + str(varneighborid) + ' is not a neighbor.')
            return None

        if np.isscalar(states):
            return states
        else:
            states = np.array(states)
            states = states / np.sum(states, axis=1).reshape((-1, 1))
        
        delta = np.product(states[:, 0] - states[:, 1])
        return 0.5*np.array([1+delta, 1-delta])


class CheckNodeFFT(GenericNode):
    """
    Class @class CheckNodeFFT creates a single check node within bipartite factor graph.
    This class relies on fast Fourier transform.
    """

    def __init__(self, checknodeid, messagelength, neighbors=None):
        """
        Initialize check node of type @class CheckNodeFFT.
        :param checknodeid: Unique identifier for check node
        :param messagelength: Length of incoming and outgoing messages
        :param neighbors: Neighbors of node @var checknodeid in bipartite graph
        """

        super().__init__(checknodeid, neighbors)
        # Length of messages
        self.__MessageLength = messagelength

    def reset(self):
        """
        Reset every states check node to uninformative measures (FFT of all ones)
        """
        uninformative = np.fft.rfft(np.ones(self.__MessageLength, dtype=float))
        # The length of np.fft.rfft is NOT self.__MessageLength.
        for neighborid in self.neighbors:
            self.setstate(neighborid, uninformative)

    def setmessagefromvar(self, varneighborid, message):
        """
        Incoming message from variable node neighbor @var vaneighborid to check node self.
        :param varneighborid: Variable node identifier of origin
        :param message: Incoming belief vector
        """
        self.setstate(varneighborid, np.fft.rfft(message))

    def getmessagetovar(self, varneighborid):
        """
        Outgoing message from check node self to variable node @var varneighbor
        :param varneighborid: Variable node identifier of destination
        :return: Outgoing belief vector
        """
        dictionary = self.getstates()
        if varneighborid is None:
            states = list(dictionary.values())
        elif varneighborid in dictionary:
            states = [dictionary[key] for key in dictionary if key is not varneighborid]
        else:
            print('Destination variable node ID ' + str(varneighborid) + ' is not a neighbor.')
            return None
        if np.isscalar(states):
            return states
        else:
            states = np.array(states)
            if states.ndim == 1:
                outgoing_fft = states
            elif states.ndim == 2:
                try:
                    outgoing_fft = np.prod(states, axis=0)
                except ValueError as e:
                    print(e)
                    return None
            else:
                raise RuntimeError('states.ndim = ' + str(np.array(states).ndim) + ' is not allowed.')
            outgoing = np.fft.irfft(outgoing_fft, axis=0)
        # The outgoing message values should be indexed using the modulo operation.
        # This is implemented by retaining the value at zero and flipping the order of the remaining vector
        # That is, for n > 0, outgoing[-n % m] = np.flip(outgoing[1:])[n]
        outgoing[1:] = np.flip(outgoing[1:])  # This is implementing the required modulo operation
        return outgoing


# class CheckNodeFWHT(GenericNode):
#     """
#     Class @class CheckNodeFWHT creates a single check node within bipartite factor graph.
#     This class relies on fast Walsh-Hadamard transform.
#     """

#     def __init__(self, checknodeid, messagelength, neighbors=None):
#         """
#         Initialize check node of type @class CheckNodeFWHT.
#         :param checknodeid: Unique identifier for check node
#         :param messagelength: Length of incoming and outgoing messages
#         :param neighbors: Neighbors of node @var checknodeid in bipartite graph
#         """

#         super().__init__(checknodeid, neighbors)
#         # Length of messages
#         self.__MessageLength = messagelength

#     def reset(self):
#         """
#         Reset every states check node to uninformative measures (FWHT of all ones)
#         """
#         uninformative = np.ones(self.__MessageLength, dtype=float)
#         # @method fht acts in place on np.array of @type float
#         ffht.fht(uninformative)
#         for neighborid in self.neighbors:
#             self.setstate(neighborid, uninformative)

#     def setmessagefromvar(self, varneighborid, message):
#         """
#         Incoming message from variable node neighbor @var vaneighborid to check node self.
#         :param varneighborid: Variable node identifier of origin
#         :param message: Incoming belief vector
#         """
#         message = message.astype(float)
#         # @method fht acts in place on np.array of @type float
#         ffht.fht(message)
#         self.setstate(varneighborid, message)

#     def getmessagetovar(self, varneighborid):
#         """
#         Outgoing message from check node self to variable node @var varneighbor
#         :param varneighborid: Variable node identifier of destination
#         :return: Outgoing belief vector
#         """
#         dictionary = self.getstates()
#         if varneighborid is None:
#             states = list(dictionary.values())
#         elif varneighborid in dictionary:
#             states = [dictionary[key] for key in dictionary if key is not varneighborid]
#         else:
#             print('Destination variable node ID ' + str(varneighborid) + ' is not a neighbor.')
#             return None
#         if np.isscalar(states):
#             return states
#         else:
#             states = np.array(states)
#             if states.ndim == 1:
#                 outgoing_fwht = states
#             elif states.ndim == 2:
#                 try:
#                     outgoing_fwht = np.prod(states, axis=0)
#                 except ValueError as e:
#                     print(e)
#                     return None
#             else:
#                 raise RuntimeError('states.ndim = ' + str(np.array(states).ndim) + ' is not allowed.')
#             outgoing = outgoing_fwht.astype(float)
#             # Inversse of FWHT is, again, FWHT
#             # @method fht acts in place on np.array of @type float
#             ffht.fht(outgoing)
#         # The outgoing message values should be indexed using the minus operation.
#         # This is unnecessary over this particular field, since the minus is itself.
#         return outgoing


class BipartiteGraph:
    """
    Class @class Graph creates bipartite factor graph for belief propagation.
    """

    def __init__(self, check2varedges, seclength):
        """
        Initialize bipartite graph of type @class Graph.
        Graph is specified by passing list of connections, one for every check node.
        The list for every check node contains the variable node identifiers of its neighbors.
        :param check2varedges: Edges from check nodes to variable nodes in list of lists format
        :param seclength: Length of incoming and outgoing messages
        """
        # Number of bits per section.
        self.__seclength = seclength
        # Length of index vector for every section.
        self.__sparseseclength = 2 ** self.seclength

        # Dictionary of identifiers and nodes for check nodes in bipartite graph.
        self.__CheckNodes = dict()
        # Dictionary of identifiers and nodes for variable nodes in bipartite graph.
        self.__VarNodes = dict()

        # Check identifier @var checknodeid=0 is reserved for the local observation at every variable node.
        for idx in range(len(check2varedges)):
            # Create check node identifier, starting at @var checknodeid = 1.
            # Identifier @var checknodeid = 0 is reserved for the local observation at every variable node.
            checknodeid = idx + 1
            # Create check nodes and add them to dictionary @var self.__CheckNodes.
            self.__CheckNodes.update({checknodeid: CheckNodeFFT(checknodeid, messagelength=self.sparseseclength)})
            # Add edges from check nodes to variable nodes.
            self.__CheckNodes[checknodeid].addneighbors(check2varedges[idx])
            # Create variable nodes and add them to dictionary @var self.__VarNodes.
            for varnodeid in check2varedges[idx]:
                if varnodeid not in self.__VarNodes:
                    self.__VarNodes[varnodeid] = VariableNode(varnodeid, messagelength=self.sparseseclength)
                else:
                    pass
                # Add edges from variable nodes to check nodes.
                self.__VarNodes[varnodeid].addneighbor(checknodeid)

    @property
    def seclength(self):
        return self.__seclength

    @property
    def sparseseclength(self):
        return self.__sparseseclength

    @property
    def checklist(self):
        return list(self.__CheckNodes.keys())

    @property
    def checkcount(self):
        return len(self.checklist)

    @property
    def varlist(self):
        return sorted(list(self.__VarNodes.keys()))

    @property
    def varcount(self):
        return len(self.varlist)

    def getchecknode(self, checknodeid):
        if checknodeid in self.checklist:
            return self.__CheckNodes[checknodeid]
        else:
            print('The retrival did not succeed.')
            print('Check node ID' + str(checknodeid))

    def getvarnode(self, varnodeid):
        if varnodeid in self.varlist:
            return self.__VarNodes[varnodeid]
        else:
            print('The retrival did not succeed.')
            print('Check node ID' + str(varnodeid))

    def reset(self):
        # Reset states at variable nodes to uniform measures.
        for varnode in self.__VarNodes.values():
            varnode.reset()
        # Reset states at check nodes to uninformative measures.
        for checknode in self.__CheckNodes.values():
            checknode.reset()

    def getobservation(self, varnodeid):
        if varnodeid in self.varlist:
            return self.getvarnode(varnodeid).getobservation()
        else:
            print('The retrival did not succeed.')
            print('Variable Node ID: ' + str(varnodeid))

    def getobservations(self):
        """
        This method returns local observations for all variable nodes in bipartite graph.
        Belief vectors are sorted according to @var varnodeid.
        :return: Array of local observations from all variable nodes
        """
        observations = np.empty((self.varcount, self.sparseseclength), dtype=float)
        idx = 0
        for varnodeid in self.varlist:
            observations[idx] = self.getvarnode(varnodeid).getobservation()
            idx = idx + 1
        return observations

    def setobservation(self, varnodeid, measure):
        if (len(measure) == self.sparseseclength) and (varnodeid in self.varlist):
            self.getvarnode(varnodeid).setobservation(measure)
        else:
            print('The assignment did not succeed.')
            print('Variable Node ID: ' + str(varnodeid))
            print('Variable Node Indices: ' + str(self.varlist))
            print('Length Measure: ' + str(len(measure)))
            print('Length Sparse Section: ' + str(self.sparseseclength))

    def updatechecks(self, checknodelist=None):
        """
        This method updates states of check nodes in @var checknodelist by performing message passing.
        Every check node in @var checknodelist requests messages from its variable node neighbors.
        The received belief vectors are stored locally.
        If no list is provided, then all check nodes in the factor graph are updated.
        :param checknodelist: List of identifiers for check nodes to be updated
        :return: List of identifiers for variable node contacted during update
        """
        if checknodelist is None:
            checknodelist = self.checklist
        elif np.isscalar(checknodelist):
            checknodelist = list([checknodelist])
        else:
            pass
        varneighborsaggregate = set()
        for checknodeid in checknodelist:
            try:
                checknode = self.getchecknode(checknodeid)
            except IndexError as e:
                print('Check node ID ' + str(checknodeid) + ' is not in ' + str(checknodelist))
                print('IndexError: ' + str(e))
                break
            varneighborlist = checknode.neighbors
            varneighborsaggregate.update(varneighborlist)
            # print('Updating State of Check ' + str(checknode.id), end=' ')
            # print('Using Variable Neighbors ' + str(varneighborlist))
            for varnodeid in varneighborlist:
                # print('\t Check Neighbor: ' + str(varnodeid))
                # print('\t Others: ' + str([member for member in varneighborlist if member is not varnodeid]))
                checknode.setmessagefromvar(varnodeid,
                                            self.getvarnode(varnodeid).getmessagetocheck(checknode.id))
        return list(varneighborsaggregate)

    def updatevars(self, varnodelist=None):
        """
        This method updates states of variable nodes in @var varnodelist by performing message passing.
        Every variable node in @var varnodelist requests messages from its check node neighbors.
        The received belief vectors are stored locally.
        If no list is provided, then all variable nodes in factor graph are updated.
        :param varnodelist: List of identifiers for variable nodes to be updated
        :return: List of identifiers for check node contacted during update
        """
        if varnodelist is None:
            varnodelist = self.varlist
        elif np.isscalar(varnodelist):
            varnodelist = list([varnodelist])
        else:
            pass
        checkneighborsaggregate = set()
        for varnodeid in varnodelist:
            try:
                varnode = self.getvarnode(varnodeid)
            except IndexError as e:
                print('Check node ID ' + str(varnodeid) + ' is not in ' + str(varnodelist))
                print('IndexError: ' + str(e))
                break
            checkneighborsaggregate.update(varnode.neighbors)
            # print('Updating State of Variable ' + str(varnode.id), end=' ')
            # print('Using Check Neighbors ' + str(varnode.neighbors))
            for checknodeid in varnode.neighbors:
                # print('\t Variable Neighbor: ' + str(neighbor))
                # print('\t Others: ' + str([member for member in varnode.neighbors if member is not neighbor]))
                checknode = self.getchecknode(checknodeid)
                measure = checknode.getmessagetovar(varnode.id)

                weight = np.linalg.norm(measure, ord=1)
                if weight != 0:
                    measure = measure / weight
                else:
                    pass
                varnode.setmessagefromcheck(checknodeid, measure)
        return list(checkneighborsaggregate)

    def getestimate(self, varnodeid):
        """
        This method returns belief vector associated with variable node @var varnodeid.
        :param varnodeid: Identifier of variable node to be queried
        :return: Belief vector from variable node @var varnodeid
        """
        return self.getvarnode(varnodeid).getestimate()

    def getestimates(self):
        """
        This method returns belief vectors for all variable nodes in bipartite graph.
        Belief vectors are sorted according to @var varnodeid.
        :return: Array of belief vectors from all variable nodes
        """
        estimates = np.empty((self.varcount, self.sparseseclength), dtype=float)
        idx = 0
        for varnodeid in self.varlist:
            estimates[idx] = self.getvarnode(varnodeid).getestimate()
            idx = idx + 1
        return estimates

    def getextrinsicestimate(self, varnodeid):
        """
        This method returns belief vector associated with variable node @var varnodeid,
        based solely on extrinsic information.
        It does not incorporate information from local observation @var checknodeid = 0.
        :param varnodeid: Identifier of the variable node to be queried
        :return:
        """
        return self.getvarnode(varnodeid).getmessagetocheck(0)

    def printgraph(self):
        for varnodeid in self.varlist:
            print('Var Node ID ' + str(varnodeid), end=": ")
            print(self.getvarnode(varnodeid).neighbors)
        for checknodeid in self.checklist:
            print('Check Node ID ' + str(checknodeid), end=": ")
            print(self.getchecknode(checknodeid).neighbors)

    def printgraphcontent(self):
        for varnodeid in self.varlist:
            print('Var Node ID ' + str(varnodeid), end=": ")
            print(self.getvarnode(varnodeid).getstates())
        for checknodeid in self.checklist:
            print('Check Node ID ' + str(checknodeid), end=": ")
            print(self.getchecknode(checknodeid).getstates())

    def decoder(self, stateestimates, count, includelikelihoods=False):  # NEED ORDER OUTPUT IN LIKELIHOOD MAYBE
        """
        This method seeks to disambiguate codewords from node states.
        Gather local state estimates from variables nodes and retain top values in place.
        Set values of other indices within every section to zero.
        Perform belief propagation and return `count` likely codewords.
        :param stateestimates: Local estimates from variable nodes.
        :param count: Maximum number of codewords returned.
        :param includelikelihoods: boolean flag of whether to return likelihoods of decoded words.
        :return: List of likely codewords.
        """

        # Resize @var stateestimates to match local measures from variable nodes.
        stateestimates.resize(self.varcount, self.sparseseclength)
        thresholdedestimates = np.zeros(stateestimates.shape)

        # Retain most likely values in every section.
        for idx in range(self.varcount):
            # Function np.argpartition puts indices of top arguments at the end (unordered).
            # Variable @var trailingtopindices holds these arguments.
            trailingtopindices = np.argpartition(stateestimates[idx], -count)[-count:]  # CHECK count or 1024
            # Retain values corresponding to top indices and zero out other entries.
            for topidx in trailingtopindices:
                thresholdedestimates[idx, topidx] = stateestimates[idx, topidx]

        # Find `count` most likely locations in every section and zero out the rest.
        # List of candidate codewords.
        recoveredcodewords = []
        # Function np.argpartition puts indices of top arguments at the end.
        # If count differs from above argument, then call np.argpartition again because top output are not ordered.
        # Indices of `count` most likely locations in root section
        trailingtopindices = np.argpartition(thresholdedestimates[0, :], -count)[-count:]
        # Iterating through evey retained location in root section
        for topidx in trailingtopindices:
            print('Root section ID: ' + str(topidx))
            # Reset graph, including check nodes, is critical for every root location.
            self.reset()
            rootsingleton = np.zeros(self.sparseseclength)
            rootsingleton[topidx] = 1 if (thresholdedestimates[0, topidx] != 0) else 0
            self.setobservation(1, rootsingleton)
            for idx in range(1, self.varcount):
                self.setobservation(idx + 1, thresholdedestimates[idx, :])

            ## This may only work for hierchical settings.

            # Start with full list of nodes to update.
            checknodes2update = set(self.checklist)
            self.updatechecks(checknodes2update)  # Update Check first
            varnodes2update = set(self.varlist)
            self.updatevars(varnodes2update)

            for iteration in range(self.maxdepth):  # Max depth
                sectionweights0 = np.linalg.norm(self.getestimates(), ord=0, axis=1)
                checkneighbors = set()
                varneighbors = set()

                # Update variable nodes and check for convergence
                self.updatevars(varnodes2update)
                for varnodeid in varnodes2update:
                    currentmeasure = self.getestimate(varnodeid)
                    currentweight1 = np.linalg.norm(currentmeasure, ord=1)
                    if np.isclose(currentweight1, np.amax(currentmeasure)):
                        varnodes2update = varnodes2update - {varnodeid}
                        checkneighbors.update(self.getvarnode(varnodeid).neighbors)
                        singleton = np.zeros(self.sparseseclength)
                        if np.isclose(currentweight1, 0):
                            pass
                        else:
                            singleton[np.argmax(currentmeasure)] = 1 if (thresholdedestimates[0, topidx] != 0) else 0
                        self.setobservation(varnodeid, singleton)
                if checkneighbors != set():
                    self.updatechecks(checkneighbors)
                # print('Variable nodes to update: ' + str(varnodes2update))

                # Update check nodes and check for convergence
                self.updatechecks(checknodes2update)
                for checknodeid in checknodes2update:
                    if set(self.getchecknode(checknodeid).neighbors).isdisjoint(varnodes2update):
                        checknodes2update = checknodes2update - {checknodeid}
                        varneighbors.update(self.getchecknode(checknodeid).neighbors)
                if varneighbors != set():
                    self.updatevars(varneighbors)
                # print('Check nodes to update: ' + str(checknodes2update))

                # Monitor progress and break, if appropriate
                newsectionweights0 = np.linalg.norm(self.getestimates(), ord=0, axis=1)
                if np.amin(newsectionweights0) == 0 or len(varnodes2update) == 0:
                    break
                else:
                    pass

            decoded = self.getcodeword().flatten()
            if not np.isscalar(self.testvalid(decoded)):
                recoveredcodewords.append(decoded)

        # Order candidates
        likelihoods = []
        for candidate in recoveredcodewords:
            isolatedvalues = np.prod((candidate, stateestimates.flatten()), axis=0)
            isolatedvalues.resize(self.varcount, self.sparseseclength)
            likelihoods.append(np.prod(np.amax(isolatedvalues, axis=1)))
        idxsorted = np.argsort(likelihoods)
        recoveredcodewords = [recoveredcodewords[idx] for idx in idxsorted[::-1]]

        if includelikelihoods:
            sortedlikelihoods = [likelihoods[idx] for idx in idxsorted[::-1]]
            return recoveredcodewords, sortedlikelihoods
        else:
            return recoveredcodewords


class Encoding(BipartiteGraph):

    def __init__(self, check2varedges, infonodeindices, seclength):
        super().__init__(check2varedges, seclength)

        paritycheckmatrix = []
        for checknodeid in self.checklist:
            row = np.zeros(self.varcount, dtype=int)
            for idx in self.getchecknode(checknodeid).neighbors:
                row[idx - 1] = 1
            paritycheckmatrix.append(row)
        paritycheckmatrix = np.array(paritycheckmatrix)
        print('Size of parity check matrix: ' + str(paritycheckmatrix.shape))
        print('Rank of parity check matrix: ' + str(np.linalg.matrix_rank(paritycheckmatrix)))

        if infonodeindices is None:
            systematicmatrix = self.eliminationgf2(paritycheckmatrix)
            print(systematicmatrix)
            self.__paritycolindices = []
            paritynodeindices = []
            for idx in range(self.checkcount):
                # Desirable indices are found in top rows of P_lu.transpose().
                # Below, columns of P_lu are employed instead of rows of P_lu.transpose().
                row = systematicmatrix[idx, :]
                jdx = np.argmax(row == 1)
                self.__paritycolindices.append(jdx)
                paritynodeindices.append(jdx + 1)
            self.__paritycolindices = sorted(self.__paritycolindices)
            self.__ParityNodeIndices = sorted(paritynodeindices)
            print('Number of parity column indices: ' + str(len(self.__paritycolindices)))

            self.__infocolindices = sorted(
                [colidx for colidx in range(self.varcount) if colidx not in self.__paritycolindices])
            infonodeindices = [varnodeid for varnodeid in self.varlist if varnodeid not in paritynodeindices]
            self.__InfoNodeIndices = sorted(infonodeindices)
        else:
            self.__InfoNodeIndices = sorted(infonodeindices)
            self.__infocolindices = [idx - 1 for idx in self.__InfoNodeIndices]
            self.__ParityNodeIndices = [varnodeid for varnodeid in self.varlist if varnodeid not in infonodeindices]
            self.__paritycolindices = [idx - 1 for idx in self.__ParityNodeIndices]

        self.__maxdepth = len(self.__ParityNodeIndices)

        print('Number of parity nodes: ' + str(len(set(self.__ParityNodeIndices))))
        self.__pc_parity = paritycheckmatrix[:, self.__paritycolindices]
        # print('Rank parity component: ' + str(np.linalg.matrix_rank(self.__pc_parity)))
        print(self.__pc_parity)

        print('Number of information nodes: ' + str(len(set(self.__InfoNodeIndices))))
        self.__pc_info = paritycheckmatrix[:, self.__infocolindices]
        # print('Rank info component: ' + str(np.linalg.matrix_rank(self.__pc_info)))
        print(self.__pc_info)

    @property
    def infolist(self):
        return self.__InfoNodeIndices

    @property
    def infocount(self):
        return len(self.infolist)

    @property
    def paritylist(self):
        return self.__ParityNodeIndices

    @property
    def paritycount(self):
        return len(self.paritylist)

    @property
    def maxdepth(self):
        return self.__maxdepth

    @maxdepth.setter
    def maxdepth(self, depth):
        self.__maxdepth = depth

    def eliminationgf2(self, paritycheckmatrix):
        idx = 0
        jdx = 0
        while (idx < self.checkcount) and (jdx < self.varcount):
            # Find index of largest element in remainder of column @var jdx
            while (np.amax(paritycheckmatrix[idx:, jdx]) == 0) and (jdx < (self.varcount - 1)):
                jdx += 1
            kdx = np.argmax(paritycheckmatrix[idx:, jdx]) + idx

            # Interchange rows @var kdx and @var idx
            row = np.copy(paritycheckmatrix[kdx])
            paritycheckmatrix[kdx] = paritycheckmatrix[idx]
            paritycheckmatrix[idx] = row

            rowidxtrailing = paritycheckmatrix[idx, jdx:]

            # Make copy of  column jdx to avoid altering its entries directly.
            coljdx = np.copy(paritycheckmatrix[:, jdx])
            # Set entry @var coljdx[idx] to 0 to avoid xoring pivot @var rowidxtrailing with itself
            coljdx[idx] = 0
            # Compute binary xor mask using outer product
            entries2flip = np.outer(coljdx, rowidxtrailing)
            # Python xor operator
            paritycheckmatrix[:, jdx:] = paritycheckmatrix[:, jdx:] ^ entries2flip

            idx += 1
            jdx += 1
        return paritycheckmatrix

    def getcodeword(self):
        """
        This method returns surviving codeword after systematic encoding and belief propagation.
        Codeword sections are sorted according to @var varnodeid.
        :return: Codeword in sections
        """
        codeword = np.empty((self.varcount, self.sparseseclength), dtype=int)
        idx = 0
        for varnodeid in self.varlist:
            block = np.zeros(self.sparseseclength, dtype=int)
            if not np.isclose(np.max(self.getestimate(varnodeid)), 0):
                block[np.argmax(self.getestimate(varnodeid))] = 1
            codeword[idx] = block
            idx = idx + 1
        return np.rint(codeword)

    def encodemessage(self, bits):
        """
        This method performs encoding based on Gaussian elimination over GF2.
        :param bits: Information bits comprising original message
        """
        if len(bits) == (self.infocount * self.seclength):
            bits = np.array(bits).reshape((self.infocount, self.seclength))
            # Container for fragmented message bits.
            # Initialize variable nodes within information node indices
            codewordsparse = np.zeros((self.varcount, self.sparseseclength))
            for idx in range(self.infocount):
                # Compute index of fragment @var varnodeid
                fragment = np.inner(bits[idx], 2 ** np.arange(self.seclength)[::-1])
                # Set sparse representation to all zeros, except for proper location.
                sparsefragment = np.zeros(self.sparseseclength, dtype=int)
                sparsefragment[fragment] = 1
                # Add sparse section to codeword.
                codewordsparse[self.__infocolindices[idx]] = sparsefragment
            for idx in range(self.paritycount):
                parity = np.remainder(self.__pc_info[idx, :] @ bits, 2)
                fragment = np.inner(parity, 2 ** np.arange(self.seclength)[::-1])
                # Set sparse representation to all zeros, except for proper location.
                sparsefragment = np.zeros(self.sparseseclength, dtype=int)
                sparsefragment[fragment] = 1
                # Add sparse section to codeword.
                codewordsparse[self.__paritycolindices[idx]] = sparsefragment
            codeword = np.array(codewordsparse).flatten()
            return codeword
        else:
            print('Length of input array is not ' + str(self.infocount * self.seclength))
            print('Number of information sections is ' + str(self.infocount))

    def encodemessageBP(self, bits):
        """
        This method performs systematic encoding through belief propagation.
        Bipartite graph is initialized: local observations for information blocks are derived from message sequence,
        parity states are set to all ones.
        :param bits: Information bits comprising original message
        """
        if len(bits) == (self.infocount * self.seclength):
            bits = np.array(bits).reshape((self.infocount, self.seclength))
            # Container for fragmented message bits.
            bitsections = dict()
            # Reinitialize factor graph to ensure there are no lingering states.
            # Node states are set to uninformative measures.
            self.reset()
            idx = 0
            # Initialize variable nodes within information node indices
            for varnodeid in self.infolist:
                # Message bits corresponding to fragment @var varnodeid.
                bitsections.update({varnodeid: bits[idx]})
                idx = idx + 1
                # Compute index of fragment @var varnodeid
                fragment = np.inner(bitsections[varnodeid], 2 ** np.arange(self.seclength)[::-1])
                # Set sparse representation to all zeros, except for proper location.
                sparsefragment = np.zeros(self.sparseseclength, dtype=int)
                sparsefragment[fragment] = 1
                # Set local observation for systematic variable nodes.
                self.setobservation(varnodeid, sparsefragment)
                # print('Variable node ' + str(varnodeid), end=' ')
                # print(' -- Observation changed to: ' + str(np.argmax(self.getobservation(varnodeid))))
            # Start with full list of check nodes to update.
            checknodes2update = set(self.checklist)
            self.updatechecks(checknodes2update)
            # Start with list of parity variable nodes to update.
            varnodes2update = set(self.varlist)
            self.updatevars(varnodes2update)
            # The number of parity variable nodes acts as upper bound.
            for iteration in range(self.paritycount):
                checkneighbors = set()
                varneighbors = set()

                # Update check nodes and check for convergence
                self.updatechecks(checknodes2update)
                for checknodeid in checknodes2update:
                    checknode = self.getchecknode(checknodeid)
                    if set(checknode.neighbors).isdisjoint(varnodes2update):
                        checknodes2update = checknodes2update - {checknode.id}
                        varneighbors.update(checknode.neighbors)
                if varneighbors != set():
                    self.updatevars(varneighbors)

                # Update variable nodes and check for convergence
                self.updatevars(varnodes2update)
                for varnodeid in varnodes2update:
                    currentmeasure = self.getestimate(varnodeid)
                    currentweight1 = np.linalg.norm(currentmeasure, ord=1)
                    if currentweight1 == 1:
                        varnodes2update = varnodes2update - {varnodeid}
                        checkneighbors.update(self.getvarnode(varnodeid).neighbors)
                if checkneighbors != set():
                    self.updatechecks(checkneighbors)

                if np.array_equal(np.linalg.norm(np.rint(self.getestimates()), ord=0, axis=1), [1] * self.varcount):
                    break

            self.updatechecks()
            # print(np.linalg.norm(np.rint(self.getestimates()), ord=0, axis=1))
            codeword = np.rint(self.getestimates()).flatten()
            return codeword
        else:
            print('Length of input array is not ' + str(self.infocount * self.seclength))

    def encodemessages(self, infoarray):
        """
        This method encodes multiple messages into codewords by performing systematic encoding
        and belief propagation on each individual message.
        :param infoarray: array of binary messages to be encoded
        """
        codewords = []
        for messageindex in range(len(infoarray)):
            codewords.append(self.encodemessageBP(infoarray[messageindex]))
        return np.asarray(codewords)

    def encodesignal(self, infoarray):
        """
        This method encodes multiple messages into a signal
        :param infoarray: array of binary messages to be encoded
        """
        signal = np.zeros(self.sparseseclength * self.varcount, dtype=float)
        for messageindex in range(len(infoarray)):
            signal = signal + self.encodemessageBP(infoarray[messageindex])
        return signal

    def testvalid(self, codeword):  # ISSUE IN USING THIS FOR NOISY CODEWORDS, INPUT SHOULD BE MEASURE
        # Reinitialize factor graph to ensure there are no lingering states.
        # Node states are set to uninformative measures.
        self.reset()
        if (len(codeword) == (self.varcount * self.sparseseclength)) and (
                np.linalg.norm(codeword, ord=0) == self.varcount):
            sparsesections = codeword.reshape((self.varcount, self.sparseseclength))
            # Container for fragmented message bits.
            idx = 0
            for varnodeid in self.varlist:
                # Sparse section corresponding to @var varnodeid.
                self.setobservation(varnodeid, sparsesections[idx])
                idx = idx + 1
                # print('Variable node ' + str(varnodeid), end=' ')
                # print(' -- Observation changed to: ' + str(np.argmax(self.getobservation(varnodeid))))

            # Check if all variable nodes remain sparse.
            self.updatechecks()
            self.updatevars()
            if np.array_equal(np.rint(self.getestimates()).flatten(), self.getobservations().flatten()):
                # print('Codeword is consistent.')
                return self.getcodeword()
            else:
                # print('Codeword has issues.')
                # print(np.sum(self.getobservations(), axis=1))
                # print(np.sum(self.getestimates(), axis=1))
                # print(np.sum(self.getobservations() - self.getestimates(), axis=1))
                return -1
        else:
            # print(np.linalg.norm(np.rint(self.getestimates()).flatten(), ord=0))
            # print(np.linalg.norm(self.getobservations().flatten(), ord=0))
            # print('Codeword has issues.')
            return -1


def numbermatches(codewords, recoveredcodewords, maxcount=None):
    """
    Counts number of matches between `codewords` and `recoveredcodewords`.
    CHECK: Does not insure uniqueness.
    :param codewords: List of true codewords.
    :param recoveredcodewords: List of candidate codewords from most to least likely.
    :return: Number of true codewords recovered.
    """
    # Provision for scenario where candidate count is smaller than codeword count.
    if maxcount is None:
        maxcount = min(len(codewords), len(recoveredcodewords))
    else:
        maxcount = min(len(codewords), len(recoveredcodewords), maxcount)
    matchcount = 0
    for candidateindex in range(maxcount):
        candidate = recoveredcodewords[candidateindex]
        # print('Candidate codeword: ' + str(candidate))
        # print(np.equal(codewords,candidate).all(axis=1)) # Check if candidate individual codewords
        matchcount = matchcount + (np.equal(codewords, candidate).all(axis=1).any()).astype(int)  # Check if matches any
    return matchcount
