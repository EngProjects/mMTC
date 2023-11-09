"""
State Evolution for SRLDPC Codes
"""

import numpy as np

class CheckNode():
    """
    Class for single check node that implements message passing on factor graph.

        Attributes:
            __edges (ndarray): edges connected to variable node
            __neighbors (ndarray): variable nodes connected to check node
            __idx_edge (int): index of current edge
            q (int): field size

        Methods:
            add_connection(edge, neighbor, weight): add connection to check node
            send_messages(messages, int_to_log, log_to_int): send messages
    """

    def __init__(self, num_edges, q):
        """
        Initialize CheckNode class. 

            Parameters:
                num_edges (int): number of connected edges
                q (int): field size

            Returns: 
                 <none>
        """
        self.__edges = np.zeros(num_edges, dtype=int)
        self.__neighbors = np.zeros(num_edges, dtype=int)
        self.__idx_edge = 0
        self.q = q

    def add_connection(self, edge, neighbor):
        """
        Add connection to check node. 

            Parameters:
                edge (int): index of edge connected to check node
                neighbor (int): id of connected neighbor
            
            Returns:
                 <none>
        """
        self.__edges[self.__idx_edge] = edge
        self.__neighbors[self.__idx_edge] = neighbor
        self.__idx_edge += 1

    def send_messages(self, messages):
        """
        Send messages as part of BP decoding. 

            Parameters:
                en0_messages (ndarray): array of current factor graph messages

            Returns:
                 <none>
        """
        in_msgs = messages[self.__edges].copy()
        num_edges = len(in_msgs)
        out_msgs = np.zeros(num_edges)
        
        q = self.q
        n = num_edges - 1

        cum_product = np.prod(in_msgs - 1/q)
        for i, en0 in enumerate(in_msgs):
            term_to_exclude = en0 - 1/q
            out_msgs[i] = 1/q + (q/(q-1))**(n-1) * (cum_product/term_to_exclude)
        
        messages[self.__edges] = out_msgs.copy()


class VariableNode():
    """
    Class for single variable node for message passing on factor graph.

        Attributes:
            __edges (ndarray): edges connected to variable node
            __idx_edge (int): index of current edge
            __ea0_ch (float): E[alpha_0]
            q (int): field size
        
        Methods:
            add_connection(edge): add connection
            set_ea0_ch(ea0_channel): set local observation
            get_ea0_val(tau2vals, ea0vals, tau2): convert tau2 -> E[alpha(0)]
            get_tau2_val(tau2vals, ea0vals, ea0): convert E[alpha(0)] -> tau2
            get_mse(messages, tau2vals, ea0vals): estimate output MSE
            send_messages(messages, tau2vals, ea0vals, idx_iter): send MSE messages
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
        self.__edges = np.zeros(num_edges, dtype=int)
        self.__idx_edge = 0
        self.__ea0_ch = 0.0
        self.q = q

    def add_connection(self, edge):
        """
        Add connection to variable node.

            Parameters:
                edge (int): index of edge connected to variable node
            
            Returns:
                 <none>
        """
        self.__edges[self.__idx_edge] = edge
        self.__idx_edge += 1

    def set_ea0_ch(self, ea0_channel):
        """
        Set local observation.

            Parameters:
                en0_channel (ndarray): local observation
            
            Returns:
                 <none>
        """
        self.__ea0_ch = ea0_channel

    def get_ea0_val(self, tau2vals, ea0vals, tau2):
        """
        Convert from tau^2 to E[alpha(0)]. 

            Inputs:
                tau2vals (ndarray): array of tau2 values
                ea0vals (ndarray): array of ea0vals corresponding to tau2vals
                tau2 (float): tau2 value to convert to E[alpha(0)] value

            Returns:
                ea0 (float): E[alpha(0)] value corresponding to tau2 value
        """
        idx_tau2 = np.argmin(np.abs(tau2vals - tau2))
        return ea0vals[idx_tau2]

    def get_tau2_val(self, tau2vals, ea0vals, ea0):
        """
        Convert from E[alpha(0)] to tau^2

            Inputs:
                tau2vals (ndarray): array of tau2 values
                ea0vals (ndarray): array of ea0vals corresponding to tau2vals
                ea0 (float): E[alpha(0)] to convert into a tau^2 value

            Returns:
                tau2 (float): tau2 value corresponding to E[alpha(0)] value
        """
        idx_el0 = np.argmin(np.abs(ea0vals - ea0))
        return tau2vals[idx_el0]   
    
    def get_mse(self, messages, tau2vals, ea0vals):
        """
        Compute approximate section MSE
        
            Parameters:
                messages (ndarray): array of current factor graph messages
                tau2vals (ndarray): array of tau2vals
                ea0vals (ndarray): array of ea0vals corresponding to tau2vals
            
            Returns:
                mse_ht (float): approximate MSE for current section
        """
        neighbor_el0s = messages[self.__edges].copy()
        in_tau2s = np.array([self.get_tau2_val(tau2vals, ea0vals, el0) for el0 in neighbor_el0s])
        lo_tau2 = self.get_tau2_val(tau2vals, ea0vals, self.__ea0_ch)
        in_tau2s = np.hstack(( in_tau2s, np.array([lo_tau2]) ))
        denominator = np.sum(1/in_tau2s)
        new_tau2 = 1 / denominator
        return 1 - self.get_ea0_val(tau2vals, ea0vals, new_tau2)

    def send_messages(self, messages, tau2vals, ea0vals, idx_iter):
        """
        Pass messages to neighboring check nodes

            Parameters:
                messages (ndarray): array of current factor graph messages
                tau2vals (ndarray): array of tau2vals
                ea0vals (ndarray): array of ea0vals corresponding to tau2vals
                idx_iter (int): index of current BP iteration

            Returns:
                 <none>
        """
        if idx_iter == 0:
            out_en0s = self.__ea0_ch
            messages[self.__edges] = out_en0s
        else:
            num_edges = len(self.__edges)
            neighbor_el0s = messages[self.__edges].copy()
            in_tau2s = np.array([self.get_tau2_val(tau2vals, ea0vals, el0) for el0 in neighbor_el0s])
            lo_tau2 = self.get_tau2_val(tau2vals, ea0vals, self.__ea0_ch)
            in_tau2s = np.hstack(( in_tau2s, np.array([lo_tau2]) ))
            out_en0s = np.zeros(num_edges)

            for idx in range(num_edges):
                idx_to_consider = np.setdiff1d(np.arange(num_edges+1), idx)
                denominator = np.sum(1/in_tau2s[idx_to_consider])
                new_tau2 = 1 / denominator
                out_en0s[idx] = self.get_ea0_val(tau2vals, ea0vals, new_tau2)

            messages[self.__edges] = out_en0s.copy()


class GFLDPCSE():
    """
    Class for single-user LDPC code over GF(q).
    
    Implements single-user non-binary LDPC encoding and decoding. 
    This class is intended for simulating the performance of LDPC codes 
    defined in the alist format. 

        Attributes:
            N (int): length of the code
            K (int): dimension of the code
            M (int): number of checks
            q (int): field size
            messages (ndarray): array of factor graph messages
            variable_nodes (list): list of variable nodes within factor graph
            check_nodes (list): list of check nodes within factor graph

        Methods:
            reset_observations(): reset all local observations at variable nodes
            reset_messages(): reset all graph messages to uninformative measures
            reset_graph(): reset observations + graph messages
            set_observation(varidx, observation): set local observation at idx
            get_mse(): get approximate total MSE
            density_evolution(r, max_iter): pass MSE messages
            
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

        # Skip max left/right degrees
        _ = alist.readline()

        # Obtain left degrees for all variable nodes
        left_degrees = [int(x) for x in alist.readline().split()]
        assert len(left_degrees) == N
        
        # With left degrees, create message + variable node data structures
        num_edges = np.sum(left_degrees)
        self.messages = 1/q*np.ones(num_edges, dtype=float)
        self.variable_nodes = [VariableNode(left_degrees[i], q) \
                               for i in range(N)]

        # Obtain right degrees for all check nodes
        right_degrees = [int(x) for x in alist.readline().split()]
        assert len(right_degrees) == M
        assert np.sum(right_degrees) == num_edges

        # Create check nodes with given degrees
        self.check_nodes = [CheckNode(right_degrees[i], q) for i in range(M)]

        # Define connections between check and variable nodes
        self.pcm = np.zeros((self.M, self.N), dtype=int)
        idx_edge = 0
        for i in range(N):
            curline = [int(x) for x in alist.readline().split()]
            connections = curline[0::2]
            for connection in connections:
                self.variable_nodes[i].add_connection(idx_edge)
                self.check_nodes[connection-1].add_connection(idx_edge, i)
                self.pcm[connection-1, i] = 1
                idx_edge += 1

        # Load tau2 <--> E[L0] LUT
        self.tau2vals = np.hstack((np.linspace(0.002, 4.0, 10000), np.arange(5, 2500)))
        self.ea0vals = np.loadtxt('tau2_to_el0_v4.txt')
        assert len(self.tau2vals) == len(self.ea0vals)
          
    def reset_observations(self):
        """
        Reset local observations associated with each variable node.
        
            Parameters:
                 <none>
                
            Returns:
                 <none>
        """
        for varnode in self.variable_nodes:
            varnode.set_ea0_ch(1/self.q)
            
    def reset_messages(self):
        """
        Reset all messages within LDPC factor graph.
        
            Parameters:
                 <none>
            
            Returns:
                 <none>
        """
        q = self.q
        self.messages = 1/q*np.ones(self.messages.shape)

    def reset_graph(self):
        """
        Reset graph.

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
        self.variable_nodes[varidx].set_ea0_ch(observation)
        
    def get_mse(self):
        """
        Return approx MSE
        
            Parameters:
                <none>
                
            Returns:
                total_mse_ht (float): approx total MSE
        """
        mses = np.array([self.variable_nodes[i].get_mse(self.messages, self.tau2vals, self.ea0vals) \
                         for i in range(self.N)])
        return np.sum(mses)
            
    def density_evolution(self, max_iter=100):
        """
        Perform message-passing on LDPC factor graph for density evolution.

            Parameters:
                max_iter (int): maximum BP iterations to perform

            Returns:
                <none>
        """
        for idx_iter in range(max_iter):
            
            # Variable to check node messages
            for i in range(self.N):
                self.variable_nodes[i].send_messages(self.messages, self.tau2vals, self.ea0vals, idx_iter)    

            # Check to variable node messages
            for i in range(self.M):
                self.check_nodes[i].send_messages(self.messages)
