# -*- coding: utf-8 -*-
import numpy as np
import numpy.matlib


class PolarCode():
    def __init__(self, n, k, N):
        self.n = n
        self.k = k
        self.codewords = np.zeros((N, n))
        self.R = k / n
        self.N = int(np.log2(n))
        self.LLRs = np.zeros(self.n)
        # --- Import Reliability  Sequence
        if n <= 64:
            Q = np.flip(np.array(
                [30, 32, 40, 44, 46, 47, 48, 52, 54, 55, 56, 58, 59, 60, 61, 62, 63, 64, 31, 28, 57, 24, 50, 53, 51, 45,
                 43, 16,
                 42, 29, 39, 27, 38, 26, 23,
                 36, 22, 49, 15, 20, 41, 14, 37, 12, 25, 8, 34, 21, 35, 19, 13, 18, 11, 6, 7, 10, 33, 4, 17, 9, 1, 5, 2,
                 3])) - 1
        else:
            Q = np.flip(np.genfromtxt("Q" + str(n) + ".csv", delimiter=',')) - 1

        self.Q = Q[Q < n].astype(int)
        self.frozenPos = self.Q[0:n - self.k]
        self.msgPos = self.Q[n - self.k:n]

    def encoder(self, msg, frozenValues, t):

        # --- Initialization
        codeword = np.zeros(self.n)  # Create an array to work during the encoding
        m = 2  # Bits to combine

        # --- Check if the values of the frozen bits are all 1, all 0 or random
        if (isinstance(frozenValues, int)):
            # --- Fixed number 1 or 0
            if (frozenValues == 1):
                # All values are 1
                frozenValues = np.zeros(self.n - self.k) + 1
            else:
                frozenValues = np.zeros(self.n - self.k)

        # Define the first bits of the codeword
        codeword[self.msgPos] = msg
        codeword[self.frozenPos] = frozenValues

        # --- Encode
        # Tree Structure
        # For each level
        for d in range(int(np.log2(self.n))):

            # Combine bits at current depth
            for i in range(0, self.n, m):
                # First part of u
                a = codeword[i:i + int(m / 2)]

                # Second part of u
                b = codeword[i + int(m / 2):i + m]

                # Do the addition modulo 2
                c = np.mod(a + b, 2)
                codeword[i:i + m] = np.append([c], [b], axis=1)


            m *= 2
        if (t >= 0):
            self.codewords[t, :] = codeword
        return codeword, frozenValues


    # =====================================  Decoder =========================== #
    def listDecoder(self, y, frozenValues, nL):
        # --- Initiazations
        L = np.zeros((self.n, nL, self.N + 1))  # beliefs for all decoders
        uHat = 2 * np.ones((self.n, nL, self.N + 1))  # Decisions in nL decoders
        PML = np.Inf * np.ones(nL)  # Path metric
        PML[0] = 0
        stateVec = np.zeros((2 * self.n - 1))  # State vector, common for all  decoders
        L[:, :, 0] = np.matlib.repmat(y, nL, 1).T  # belief for root
        node, depth, done = 0, 0, 0


        while (done == 0):  # Traversal loop

            # ==================== Leaf node  ==================== #
            # --- Check if you are on the leaf
            if (depth == self.N):
                DM = np.squeeze(L[node, :, self.N])  # Decision metric for all decoders
                # --- Check if it is frozen
                if (node in self.frozenPos):
                    uHat[node, :, self.N] = frozenValues[np.where(self.frozenPos == node)]
                    if (frozenValues[np.where(self.frozenPos == node)] > 0):
                        PML = PML + DM
                    else:
                        PML = PML + np.multiply(abs(DM), 1 * (DM < 0))

                else:
                    dec = 1 * (DM < 0)
                    PM2 = np.concatenate((PML, PML + abs(DM)))
                    pos, PML = minK(PM2, nL)
                    pos1 = 1 * (pos > (nL - 1))
                    idxPos = np.nonzero(pos1)
                    pos[idxPos] = pos[idxPos] - nL
                    dec = dec[pos]
                    dec[idxPos] = 1 - dec[idxPos]
                    L = L[:, pos, :]
                    uHat = uHat[:, pos, :]
                    uHat[node, :, self.N] = dec

                if (node == self.n - 1):
                    done = 1
                else:
                    node = int(np.floor(node / 2))
                    depth = int(depth - 1)

            # ==================== Other node  ==================== #
            else:
                # Find the Node position
                npos = int(2 ** depth + node)

                # ==================== L Step  ==================== #
                # ---  If it is the first time that you hit a node
                if (stateVec[npos] == 0):

                    temp = int(2 ** (self.N - depth))
                    # --- Incoming Beliefs
                    Ln = np.squeeze(L[temp * node:temp * (node + 1), :, depth]).T

                    # --- Break beliefs into two
                    a = Ln[:, 0:int(temp / 2)]
                    b = Ln[:, int(temp / 2)::]

                    node = int(2 * node)
                    depth = int(depth + 1)

                    # --- Incoming belief length for left child
                    temp = int(temp / 2)

                    # --- Sum - Product
                    L[temp * node:temp * (node + 1), :, depth] = sumProd(a, b).T


                    stateVec[npos] = 1


                else:
                    # ==================== R Step  ==================== #
                    if (stateVec[npos] == 1):

                        temp = int(2 ** (self.N - depth))
                        # --- Incoming Beliefs
                        Ln = np.squeeze(L[temp * node:temp * (node + 1), :, depth]).T

                        # --- Incoming msg
                        nodeL = int(2 * node)
                        dL = int(depth + 1)
                        tempL = int(temp / 2)
                        uHatn = np.squeeze(uHat[tempL * nodeL:tempL * (nodeL + 1), :, dL])

                        # --- Repetition decoding
                        a = Ln[:, 0:int(temp / 2)]
                        b = Ln[:, int(temp / 2)::]
                        rept = g(a.T, b.T, uHatn)

                        # --- Move to the next node
                        # Next child left
                        node = int(2 * node + 1)
                        # Next depth
                        depth = int(depth + 1)
                        # Incoming beliefs length
                        temp = int(temp / 2)
                        # Change the state of the node
                        stateVec[npos] = 2

                        # --- Save Beliefs
                        L[temp * node:temp * (node + 1), :, depth] = rept

                    # ==================== U Step  ==================== #
                    else:
                        temp = int(2 ** (self.N - depth))
                        nodeL = int(2 * node)
                        nodeR = int(2 * node + 1)
                        dC = int(depth + 1)
                        tempC = int(temp / 2)

                        # --- Incoming decisions from the left child
                        uHatL = np.squeeze(uHat[tempC * nodeL:tempC * (nodeL + 1), :, dC])

                        # --- Incoming decisions from the right child
                        uHatR = np.squeeze(uHat[tempC * nodeR:tempC * (nodeR + 1), :, dC])

                        uHat[temp * node:temp * (node + 1), :, depth] = np.vstack((((uHatL + uHatR) % 2), uHatR))

                        # --- Go back to parent
                        node = int(np.floor(node / 2))
                        depth = int(depth - 1)
        return uHat[self.msgPos, :, self.N].T, PML

    # ===================================== Functions for Decoders =========================== #



def sumProd(a, b):
    result = np.zeros((a.shape))

    max = np.maximum(abs(a),abs(b))

    r,c = np.where(max < 38)
    temp = np.multiply(np.tanh(a[r,c]/2.0), np.tanh(b[r,c]/2.0))
    temp[np.where(temp > 0.99)] = 0.99
    result[r,c] = 2*np.arctanh(temp)

    r, c = np.where(max >= 38)
    result[r,c] = minSum(a[r,c], b[r,c])

    return result



def g(a, b, c):
    return b + np.multiply((1 - 2 * c), a)


def minK(a, k):
    idx = (a).argsort()[:k]
    return idx, a[idx]



def minSum(a, b):
    # --- Sign
    signA, signB = 1 - 2 * (1 * (a < 0)), 1 - 2 * (1 * (b < 0))
    sign = np.multiply(signA, signB)
    # --- Magnitude
    magn = np.minimum(abs(a), abs(b))

    return np.multiply(magn, sign)



