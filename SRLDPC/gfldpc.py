"""
GF(2^j) LDPC encoder and message-passing based decoder
"""

import numpy as np
from pyfht import fht


class CheckNode():
    """
    Class for single check node that implements message passing on factor graph.

        Attributes:
            __edges (ndarray): edges connected to variable node
            __neighbors (ndarray): variable nodes connected to check node
            __weights (ndarray): weights of connected edges
            __idx_edge (int): index of current edge

        Methods:
            add_connection(edge, neighbor, weight): add connection to check node
            send_messages(messages, int_to_log, log_to_int): send messages
            check_consistency(estimate, int_to_log, log_to_int): chk consistency 
    """

    def __init__(self, num_edges):
        """
        Initialize CheckNode class. 

            Parameters:
                num_edges (int): number of connected edges

            Returns: 
                 <none>
        """
        self.__edges = np.zeros(num_edges, dtype=int)
        self.__neighbors = np.zeros(num_edges, dtype=int)
        self.__weights = np.zeros(num_edges, dtype=int)
        self.__idx_edge = 0

    def add_connection(self, edge, neighbor, weight):
        """
        Add connection to check node. 

            Parameters:
                edge (int): index of edge connected to check node
                neighbor (int): id of connected neighbor
                weight (int): weight corresponding to edge
            
            Returns:
                 <none>
        """
        self.__edges[self.__idx_edge] = edge
        self.__neighbors[self.__idx_edge] = neighbor
        self.__weights[self.__idx_edge] = weight
        self.__idx_edge += 1

    def send_messages(self, messages, int_to_log, log_to_int):
        """
        Send messages as part of BP decoding. 

            Parameters:
                messages (ndarray): array of current factor graph messages
                int_to_log (ndarray): LUT converting int to log in GF(q)
                log_to_int (ndarray): LUT converting log to int in GF(q)

            Returns:
                 <none>
        """
        in_msgs = messages[self.__edges, :].copy()
        num_edges, q = in_msgs.shape
        c = 1 / (np.sqrt(2)**int(np.log2(q)))
        
        # FHT of incoming messages
        for i in range(num_edges):
            in_msgs[i, :] = fht(in_msgs[i, :])
            
        # Compute \overline{\mu}_{c \rightarrow v} according to Remark 8
        out_msgs = np.zeros(in_msgs.shape, dtype=float)
        for i in range(num_edges):
            indices = np.setdiff1d(np.arange(num_edges), i)
            out_msgs[i, :] = np.prod(in_msgs[indices, :], axis=0)
            out_msgs[i, :] = c*fht(c*out_msgs[i, :])
            
        # Compute \mu_{c \rightarrow v} according to (21)
        logindices = int_to_log[np.arange(1, q)]
        for i in range(num_edges):
            logweight = int_to_log[self.__weights[i]]
            newlogindices = (logweight + logindices) % (q-1)
            newindices = np.hstack((0, log_to_int[newlogindices]))
            out_msgs[i, :] = out_msgs[i, newindices]
        
        messages[self.__edges, :] = out_msgs.copy()

    def check_consistency(self, estimate, int_to_log, log_to_int):
        """
        Check for parity consistency across neighboring nodes.

            Parameters:
                estimate (ndarray): hard decision cdwd estimate as GF(q) symbols
                int_to_log (ndarray): LUT converting int to log in GF(q)
                log_to_int (ndarray): LUT converting log to int in GF(q)
            
            Returns:
                is_consistent (bool): flag of whether check is satisfied
        """
        neighborhood_sum = 0
        q = len(int_to_log)
        for i in range(self.__idx_edge):
            if estimate[self.__neighbors[i]] == 0:
                continue
            logw = int_to_log[self.__weights[i]]
            logv = int_to_log[estimate[self.__neighbors[i]]]
            intwv = log_to_int[(logw + logv) % (q-1)]
            neighborhood_sum ^= intwv
        return (neighborhood_sum == 0)


class VariableNode():
    """
    Class for single variable node for message passing on factor graph.

        Attributes:
            q (int): field size
            __edges (ndarray): edges connected to variable node
            __invweights (ndarray): inverse of weights of connected edges
            __idx_edge (int): index of current edge
            _observation (ndarray): local observation for variable node
            estimate (ndarray): estimated probability distribution for variable
        
        Methods:
            add_connection(edge, weight, int_to_log, log_to_int): add connection
            set_observation(observation): set local observation
            compute_estimate(messages): compute estimated prob. dist at node
            send_messages(messages, int_to_log, log_to_int): send messages
    """

    def __init__(self, num_edges, q):
        """
        Initialize VariableNode class.

            Parameters:
                num_edges (int): number of connected edges
                q (int): field size

            Returns:
                 <none>
        """
        self.q = q
        self.__edges = np.zeros(num_edges, dtype=int)
        self.__invweights = np.zeros(num_edges, dtype=int)
        self.__idx_edge = 0
        self._observation = np.zeros(q, dtype=float)
        self.estimate = np.zeros(q, dtype=float)

    def add_connection(self, edge, weight, int_to_log, log_to_int):
        """
        Add connection to variable node.

            Parameters:
                edge (int): index of edge connected to variable node
                weight (int): weight of edge
                int_to_log (ndarray): LUT converting int to log in GF(q)
                log_to_int (ndarray): LUT converting log to int in GF(q)
            
            Returns:
                 <none>
        """
        self.__edges[self.__idx_edge] = edge
        log_weight = int_to_log[weight]
        log_inv_weight = (self.q-1-log_weight) % (self.q-1)
        self.__invweights[self.__idx_edge] = log_to_int[log_inv_weight]
        self.__idx_edge += 1

    def set_observation(self, observation):
        """
        Set local observation.

            Parameters:
                observation (ndarray): local observation
            
            Returns:
                 <none>
        """
        self._observation = observation.copy().flatten() # reshape(1, -1)
    
    def compute_estimate(self, messages):
        """
        Compute estimated probability distribution at variable node.
        
            Parameters:
                messages (ndarray): array of current factor graph messages
                
            Returns:
                estimate (ndarray): estimated prob. dist. at variable node
        """
        in_msgs = messages[self.__edges, :].copy()
        num_edges, q = in_msgs.shape
        
        # Compute estimate of probability distribution associated with var node
        self.estimate = np.prod(in_msgs, axis=0)*self._observation
        self.estimate /= (np.linalg.norm(self.estimate, ord=1) + 1e-12)
        
        return self.estimate
    
    def send_messages(self, messages, int_to_log, log_to_int):
        """
        Pass messages to neighboring check nodes

            Parameters:
                messages (ndarray): array of current factor graph messages
                int_to_log (ndarray): LUT converting int to log in GF(q)
                log_to_int (ndarray): LUT converting log to int in GF(q)

            Returns:
                 <none>
        """
        in_msgs = messages[self.__edges, :].copy()
        num_edges, q = in_msgs.shape
        
        # Compute messages \mu_{v \rightarrow c} according to (23)
        out_msgs = np.zeros(in_msgs.shape)
        for i in range(num_edges):
            indices = np.setdiff1d(np.arange(num_edges), i)
            out_msgs[i, :] = np.prod(in_msgs[indices, :], axis=0)
        out_msgs *= self._observation
        out_msgs /= (np.linalg.norm(out_msgs, axis=1, ord=1).reshape(-1, 1) + 1e-12)
        
        # Compute messages \overline{\mu}_{v \rightarrow c} according to (20)
        logindices = int_to_log[np.arange(1, q).astype(int)]
        for i in range(num_edges):
            loginvweight = int_to_log[self.__invweights[i]]
            newlogindices = (loginvweight+logindices) % (q-1)
            newindices = np.hstack((0, log_to_int[newlogindices]))
            out_msgs[i, :] = out_msgs[i, newindices]
        
        # Send messages to connected check nodes
        messages[self.__edges, :] = out_msgs.copy()


class GFLDPC():
    """
    Class for single-user LDPC code over GF(q).
    
    Implements single-user non-binary LDPC encoding and decoding. 
    This class is intended for simulating the performance of LDPC codes 
    defined in the alist format. 

        Attributes:
            int_to_log (ndarray): int->log LUT for GF(q)
            log_to_int (ndarray): log->int LUT for GF(q)
            N (int): length of the code
            K (int): dimension of the code
            M (int): number of checks
            q (int): field size
            messages (ndarray): array of factor graph messages
            variable_nodes (list): list of variable nodes within factor graph
            check_nodes (list): list of check nodes within factor graph

        Methods:
            generate_log_tables(gf_polynomial): generate int/log LUTs for GF(q)
            eliminationfq(): rref of H over GF(q) to find K info symbols
            reset_observations(): reset all local observations at variable nodes
            reset_messages(): reset all graph messages to uninformative measures
            reset_graph(): reset all local observations and graph messages
            set_observation(varidx, observation): set local observation at idx
            get_estimates(): get estimated prob. dist. at each variable node
            encode(message): encode binary message into non-binary codeword
            check_consistency(codeword_ht): chk if codeword_ht is a valid cdwd
            bp_decoder(r, max_iter): decode via belief propagation (BP)
            
        Constraints:
            field must be an extension of the binary field; i.e. q = 2**j
    """

    def __init__(self, alist_file_name):
        """
        Initialize GFLDPC class. 

            Parameters:
                alist_file_name (str): name of file with alist code definition

            Returns:
                 <none>
        """
        
        # Begin reading alist file
        alist = open(alist_file_name, 'r')

        # Obtain N, M, q for LDPC code
        [N, M, q] = [int(x) for x in alist.readline().split()] 
        self.N = N
        self.M = M
        self.K = N - M
        self.q = q
        
        # Obtain field polynomial associated with GF(q)
        gf_polynomial = self.generate_field_polynomial(q)
        
        # Create int/log LUTs associated with GF(q)
        self.int_to_log, self.log_to_int = self.generate_log_tables(gf_polynomial)

        # Skip max left/right degrees
        _ = alist.readline()

        # Obtain left degrees for all variable nodes
        left_degrees = [int(x) for x in alist.readline().split()]
        assert len(left_degrees) == N
        
        # With left degrees, create message + variable node data structures
        num_edges = np.sum(left_degrees)
        self.messages = np.ones((num_edges, q), dtype=float)
        self.variable_nodes = [VariableNode(left_degrees[i], q) \
                               for i in range(N)]

        # Obtain right degrees for all check nodes
        right_degrees = [int(x) for x in alist.readline().split()]
        assert len(right_degrees) == M
        assert np.sum(right_degrees) == num_edges

        # Create check nodes with given degrees
        self.check_nodes = [CheckNode(right_degrees[i]) for i in range(M)]

        # Define connections between check and variable nodes
        self.pcm = np.zeros((self.M, self.N), dtype=int)
        idx_edge = 0
        for i in range(N):
            curline = [int(x) for x in alist.readline().split()]
            connections = curline[0::2]
            weights = curline[1::2]
            for (connection, weight) in zip(connections, weights):
                self.variable_nodes[i].add_connection(idx_edge, weight, 
                                            self.int_to_log, self.log_to_int)
                self.check_nodes[connection-1].add_connection(idx_edge, i, weight)
                self.pcm[connection-1, i] = weight
                idx_edge += 1
                
        # Row reduce parity check matrix (pcm) for encoding
        self.eliminationfq()
        
    def generate_field_polynomial(self, q):
        """
        Generate field polynomial for field of order q
        
            Parameters:
                q (int): field size
                
            Returns: 
                gf_polynomial (ndarray): field polynomial of form [a0, ..., an]
        """
        polynomials = {
            8: np.array([1, 1, 0, 1]),
            16: np.array([1, 1, 0, 0, 1]),
            32: np.array([1, 0, 1, 0, 0, 1]),
            64: np.array([1, 1, 0, 0, 0, 0, 1]),
            128: np.array([1, 0, 0, 1, 0, 0, 0, 1]),
            256: np.array([1, 0, 1, 1, 1, 0, 0, 0, 1]),
            512: np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
            1024: np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1])
            }
        return polynomials[q]     
    
    def generate_log_tables(self, gf_polynomial):
        """
        Generate int/log LUTs associated with GF(q)
        
            Parameters:
                gf_polynomial (ndarray): field polynomial of form [a0, ..., an]
                
            Returns:
                int_to_log (ndarray): int -> log LUT
                log_to_int (ndarray): log -> int LUT
        """
        # Input validation
        px = np.flip(gf_polynomial)
        assert not (px > 1).any()
        assert not (px < 0).any()

        # Extract info from input
        m = len(px) - 1
        field_size = 2**m
        px_int = np.dot(px, 2**np.arange(len(px))[::-1]).astype(int)

        # Create log-to-int table
        log_to_int = np.ones(field_size-1, dtype=int)
        for power in range(1, field_size-1):
            log_to_int[power] = log_to_int[power-1] << 1
            if log_to_int[power] >= field_size:
                log_to_int[power] = log_to_int[power] ^ px_int

        # Create int-to-log table
        int_to_log = np.zeros(field_size, dtype=int)
        for idx in range(field_size):
            if idx == 0:
                int_to_log[idx] = -1
            else:
                int_to_log[idx] = np.where(log_to_int == idx)[0]
                
        return int_to_log, log_to_int
          
    def eliminationfq(self):
        """
        Gaussian elimination over GF(q) to find K information bits
        
            Parameters:
                 <none>
            
            Returns:
                 <none>
        """    
        idx = 0
        jdx = 0
        while (idx < self.M) and (jdx < self.N):
            # Identify pivot
            while (np.amax(self.pcm[idx:, jdx]) == 0) and (jdx < (self.N - 1)):
                jdx += 1
            kdx = np.argmax(self.pcm[idx:, jdx]) + idx

            # Interchange rows kdx and idx
            row = np.copy(self.pcm[kdx])
            self.pcm[kdx] = self.pcm[idx]
            self.pcm[idx] = row
            
            # Normalize rest of interchanged row
            element = self.pcm[idx, jdx]
            if element != 1:
                logelement = self.int_to_log[element]
                loginv = (self.q-1-logelement) % (self.q-1)
                idxnonzero = np.where(self.pcm[idx, jdx:])[0]
                logrow = self.int_to_log[self.pcm[idx, jdx+idxnonzero]]
                lognewrow = (logrow + loginv) % (self.q-1)
                self.pcm[idx, jdx+idxnonzero] = self.log_to_int[lognewrow]
            
            # Edit all rows beneath current row
            ldx = idx+1
            while ldx < self.M:
                if self.pcm[ldx, jdx] == 0:
                    pass
                else:
                    logval = self.int_to_log[self.pcm[ldx, jdx]]
                    loginv = (self.q-1-logval) % (self.q-1)
                    idxnonzero = np.where(self.pcm[ldx, jdx:])[0]
                    logrow = self.int_to_log[self.pcm[ldx, jdx+idxnonzero]]
                    lognewrow = (logrow + loginv) % (self.q-1)
                    self.pcm[ldx, jdx+idxnonzero] = self.log_to_int[lognewrow]
                    self.pcm[ldx, jdx:] ^= self.pcm[idx, jdx:]
                ldx += 1
                    
            # Increment idx, jdx
            idx += 1
            jdx += 1
            
    def reset_observations(self):
        """
        Reset local observations associated with each variable node.
        
            Parameters:
                 <none>
                
            Returns:
                 <none>
        """
        uninformative_measure = (1/self.q)*np.ones(self.q)
        for varnode in self.variable_nodes:
            varnode.set_observation(uninformative_measure)
            
    def reset_messages(self):
        """
        Reset all messages within LDPC factor graph.
        
            Parameters:
                 <none>
            
            Returns:
                 <none>
        """
        self.messages = np.ones(self.messages.shape)

    def reset_graph(self):
        """
        Reset all messages and local observations within factor graph.

            Parameters:
                 <none>

            Returns:
                 <none>
        """
        self.reset_observations()
        self.reset_messages()
            
    def set_observation(self, varidx, observation):
        """
        Set local observation at variable node idx.
            
            Parameters:
                varidx (int): index of variable node
                observation (ndarray): local observation for varnode idx
                
            Returns:
                 <none>
        """
        self.variable_nodes[varidx].set_observation(observation)
        
    def get_estimates(self):
        """
        Return estimated probability distributions at each var node.
        
            Parameters:
                <none>
                
            Returns:
                estimates (ndarray): estimated prob. dist. at each var node
        """
        estimates = np.zeros(self.N*self.q)
        for i in range(self.N):
            estimates[i*self.q:(i+1)*self.q] = self.variable_nodes[i] \
                                               .compute_estimate(self.messages)
        return estimates
        
    def encode(self, message):
        """
        Encode binary or non-binary message into non-binary LDPC codeword
        
            Parameters:
                message (ndarray): information message to be encoded
            
            Returns:
                codeword (ndarray): codeword corresponding to info msg
        """
        bits_per_symbol = int(np.log2(self.q))
        msg_symbols = np.zeros(self.K).astype(int)
        
        # If message is binary, convert to GF(q) symbols
        if len(message) == self.K*bits_per_symbol:
            for i in range(self.K):
                msg_bits = message[i*bits_per_symbol:(i+1)*bits_per_symbol]
                symbol = np.dot(msg_bits, 2**np.arange(bits_per_symbol)[::-1])
                msg_symbols[i] = symbol
        elif len(message) == self.K:
            msg_symbols = message
        else:
            raise RuntimeError('Cannot encode message of length ' + \
                               str(len(message)))
            
        # Encode GF(q) symbols into valid codeword
        codeword = np.zeros(self.N).astype(int)
        codeword[self.M:] = msg_symbols.copy()
        for i in range(self.M):
            idx = self.M-1-i
            weights = self.pcm[idx, idx+1:]
            cdwd_i = codeword[idx+1:]
            for j in range(self.K+i):
                if weights[j] == 0 or cdwd_i[j] == 0:
                    continue
                logsymb = self.int_to_log[cdwd_i[j]]
                logweight = self.int_to_log[weights[j]]
                logprod = (logsymb+logweight) % (self.q-1)
                weighted_symbol = self.log_to_int[logprod]
                codeword[idx] ^= weighted_symbol 
        
        return codeword

    def check_consistency(self, codeword_ht):
        """
        Check to see if codeword_ht satisfies all parity checks.
        
            Parameters:
                codeword_ht (ndarray): candidate codeword
                
            Returns:
                is_consistent (bool): flag of whether codeword is valid
        """
        for i in range(self.M):
            consistent = self.check_nodes[i].check_consistency(codeword_ht, \
                                            self.int_to_log, self.log_to_int)
            if not consistent:
                return False
        return True
            
    def bp_decoder(self, max_iter=100):
        """
        Perform message-passing on LDPC factor graph to decode codeword.

            Parameters:
                max_iter (int): maximum BP iterations to perform

            Returns:
                cdwd_ht (ndarray): estimate of true codeword
        """
        for idx_iter in range(max_iter):
            
            # Variable to check node messages
            for i in range(self.N):
                self.variable_nodes[i].send_messages(self.messages,  \
                                                     self.int_to_log, \
                                                     self.log_to_int)      

            # Check to variable node messages
            for i in range(self.M):
                self.check_nodes[i].send_messages(self.messages,  \
                                                  self.int_to_log, \
                                                  self.log_to_int)
                        
            # Make hard decisions on current codeword estimate 
            cdwd_ht = np.array([np.argmax(varnode.compute_estimate(self.messages)) 
                                for varnode in self.variable_nodes])
            
            # End iterations early if codeword esimate is parity-consistent
            if self.check_consistency(cdwd_ht): 
                # if max_iter > 1: print('breaking at iteration ' + str(idx_iter))
                break
