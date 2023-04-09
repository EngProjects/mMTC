# -*- coding: utf-8 -*-
import numpy as np
from PolarCode import PolarCode
from utilities import bin2dec, dec2bin, crcEncoder, modQPSK

class UEs:
    def __init__(self, nUEs, nAPs, nAnts, B, Bf, L, nc, nL, nPilots, sigma2, Pt):
        ''' Parameters '''
        self.nUEs = nUEs  # Number of Users
        self.nAPs = nAPs  # Number of APs
        self.nAnts = nAnts # Number of antennas per APs
        self.totalAnt = self.nAnts * self.nAPs
        self.Bf = Bf  # number of bits of the first part of the message
        self.Bs = B - Bf  # number of bits of the second part of the message
        self.L = L  # Length of spreading sequence
        self.J = 2 ** Bf  # Number of spreading sequence
        self.nc = nc  # length of code
        self.nL = nL  # List size
        self.nQPSKSymbols = int(nc / 2)  # Number of QPSK symbols
        self.nDataSymbols = int(L * self.nQPSKSymbols)
        self.nPilots = nPilots  # number of pilot symbols
        self.nChanlUses = self.nPilots + self.nDataSymbols
        self.sigma2 = sigma2
        self.Pt = Pt

        ''' For polar code '''
        # Polynomial for CRC coding
        self.divisor = np.array([1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1], dtype=int)

        self.lCRC = len(self.divisor)  # Number of CRC bits
        self.msgLen = self.Bs + self.lCRC  # Length of the input to the encoder
        self.frozenValues = np.zeros((self.nc - self.msgLen, self.J))

        # Create a polar Code object
        self.polar = PolarCode(self.nc, self.msgLen, self.nUEs)

        ''' Generate matrices '''
        # Pilots
        self.P = np.sqrt(self.Pt/2) * ((1 - 2 * np.round(np.random.randint(low=0, high=2, size=(self.nPilots, self.J)))) + 1j * (
                1 - 2 * np.round(np.random.randint(low=0, high=2, size=(self.nPilots, self.J)))))

       
        # Spreading sequence master set
        self.A = (np.random.normal(loc=0, scale=1, size=(self.L, self.J)) + 1j * np.random.normal(loc=0, scale=1, size=(self.L, self.J)))

        temp = np.linalg.norm(self.A, axis=0)
        self.A = np.divide(self.A, temp)

        self.A = (np.sqrt(self.L) * self.A) * np.sqrt(self.Pt)


        ''' To store information '''
        self.msgs = np.zeros((nUEs, Bf + self.Bs), dtype=int)  # Store the active messages
        self.msgsHat = np.zeros((nUEs, Bf + self.Bs), dtype=int)  # Store the recovered messages

        self.idxSS = np.zeros(self.nUEs, dtype=int)

        self.count = 0  # Count the number of recovered msgs in this round
        self.Y = np.zeros((self.nChanlUses, nAPs))
        self.idxSSDec = np.array([], dtype=int)
        self.idxSSHat = np.array([], dtype=int)  # To store the new recovered sequences
        self.symbolsHat = np.zeros((self.nUEs, self.nQPSKSymbols), dtype=complex)
        self.codewords = np.zeros((self.nUEs, self.nc))

        self.idx2UE = np.zeros(self.J, dtype=int) - 1
        self.UE2idx = np.zeros(self.nUEs, dtype=int) - 1


    def transmit(self, msgs, H):

        '''
        Function to encode the messages of the users
        Inputs: 1. the message of the users in the binary form, dimensions of msgBin, K x B
                2. Channel
        Output: The sum of the channel output before noise, dimensions of Y, n x nAPs
        '''

        # ===================== Initialization ===================== #
        HX = np.zeros((self.nChanlUses, self.totalAnt), dtype=complex)

        # --- Step 0: Save the messages
        self.msgs = msgs.copy()
        # --- For all active users
        for k in range(self.nUEs):

            # --- Step 1: Break the message into two parts
            # First part, Second part
            mf = self.msgs[k, 0:self.Bf]
            ms = self.msgs[k, self.Bf::]

            # --- Step 2: Find the decimal representation of mf
            self.idxSS[k] = bin2dec(mf)
            self.idx2UE[self.idxSS[k]] = int(k)
            self.UE2idx[k] = int(self.idxSS[k])

            # --- Step 3: Append CRC bits to ms
            msgCRC = crcEncoder(ms, self.divisor)

            # --- Step 4: polar encode
            codeword, _ = self.polar.encoder(msgCRC, self.frozenValues[:, self.idxSS[k]], k)
            self.codewords[k,:] = codeword

            # --- Step 5: QPSK modulation
            symbols = modQPSK(codeword)

            # --- For Pilots (PH)
            PH = np.kron(self.P[:, self.idxSS[k]], H[k, :]).reshape(self.nPilots, self.totalAnt)

            # --- For Symbols (QH)
            A = np.kron(symbols, self.A[:, self.idxSS[k]])
            QH = np.kron(A, H[k, :]).reshape(self.nDataSymbols, self.totalAnt)

            # --- Add the new matrix to the output signal
            HX += np.vstack((PH, QH))


        return HX

