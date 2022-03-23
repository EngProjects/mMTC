# -*- coding: utf-8 -*-

import numpy as np
from PolarCode import PolarCode
from utility import bin2dec, dec2bin, crcEncoder, crcDecoder

from  scipy import linalg

class FASURA():
    def __init__(self, K, nPilots, B, Bf, L, nc, nL, M, sigma2):
        self.K = K
        self.Bf = Bf  # number of bits of the first part of the message
        self.Bs = B - Bf  # number of bits of the second part of the message
        self.L = L  # Length of spreading sequence
        self.J = 2 ** Bf  # Number of spreading sequence
        self.nc = nc  # length of code
        self.nL = nL  # List size
        self.nChanlUses = int((nc / np.log2(4)) * L + nPilots)
        self.nDataSlots = int(nc / np.log2(4))
        self.M = M  # Number of antennas
        self.nPilots = nPilots  # number of pilot symbols




        # Pilots
        self.P = ((1 - 2 * np.round(np.random.randint(low=0, high=2, size=(self.nPilots, self.J)))) + 1j * (
                1 - 2 * np.round(np.random.randint(low=0, high=2, size=(self.nPilots, self.J))))) / np.sqrt(2.0)

        # spreading sequence master set
        self.A = (np.random.normal(loc=0, scale=1, size=(self.nDataSlots * self.L, self.J)) + 1j * np.random.normal(
            loc=0, scale=1, size=(self.nDataSlots * self.L, self.J)))

        for j in range(self.nDataSlots):
            temp = np.linalg.norm(self.A[j * self.L:(j + 1) * self.L, :], axis=0)
            self.A[j * self.L:(j + 1) * self.L, :] = np.divide(self.A[j * self.L:(j + 1) * self.L, :], temp)

        self.A = (np.sqrt(self.L) * self.A)

        # Polynomial for CRC coding
        if K <= 100:
            self.divisor = np.array([1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1], dtype=int)
        else:
            self.divisor = np.array([1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1], dtype=int)

        self.lCRC = len(self.divisor)  # Number of CRC bits
        self.msgLen = self.Bs + self.lCRC  # Length of the input to the encoder
        self.frozenValues = np.round(np.random.randint(low=0, high=2, size=(nc - self.msgLen, self.J)))

        # Create a polar Code object
        self.polar = PolarCode(nc, self.msgLen, K)

        self.sigma2 = sigma2  # variance of the noise
        self.interleaver = np.zeros((self.nc, self.J), dtype=int)
        for j in range(self.J):
            self.interleaver[:, j] = np.random.choice(self.nc, self.nc, replace=False)


        self.msgs = np.zeros((K, Bf + self.Bs))  # Store the active messages
        self.msgsHat = np.zeros((K, Bf + self.Bs))  # Store the recovered messages
        self.count = 0  # Count the number of recovered msgs in this round
        self.Y = np.zeros((self.nChanlUses, M))
        self.idxSSDec = np.array([], dtype=int)
        self.idxSSHat = np.array([], dtype=int)  # To store the new recovered sequences
        self.symbolsHat = np.zeros((self.K, self.nDataSlots), dtype=complex)
        self.NOPICE = 1





    def transmitter(self, msgBin, H):

        '''
        Function to encode the messages of the users
        Inputs: 1. the message of the users in the binary form, dimensions of msgBin, K x B
                2. Channel
        Output: The sum of the channel output before noise, dimensions of Y, n x M
        '''


        # ===================== Initialization ===================== #
        Y = np.zeros((self.nChanlUses, self.M), dtype=complex)

        # --- For all active users
        for k in range(self.K):

            # --- Save the active message k
            self.msgs[k, :] = msgBin[k, :]


            # --- Break the message into to 2 parts
            # First part, Second part
            mf = msgBin[k, 0:self.Bf]
            ms = msgBin[k, self.Bf::]

            # --- Find index of the spreading sequence
            idxSS = bin2dec(mf)

            # --- Add CRC
            msgCRC = crcEncoder(ms, self.divisor)

            # --- polar encoder
            codeword, _ = self.polar.encoder(msgCRC, self.frozenValues[:, idxSS], k)

            # --- Interleaver
            codeword = codeword[self.interleaver[:, idxSS]]

            # --- QPSK modulation
            symbols = QPSK(codeword)


            # --- Initialize two temp Matrices
            YTempPilots = np.zeros((self.nPilots, self.M), dtype=complex)
            YTempSymbols = np.zeros((self.nDataSlots * self.L, self.M), dtype=complex)

            # --- For Pilots (PH)
            for m in range(self.M):
                YTempPilots[:, m] = self.P[:, idxSS] * H[k, m]

            # --- For Symbols (QH)
            A = np.zeros((self.nDataSlots * self.L), dtype=complex)
            for t in range(self.nDataSlots):
                A[t * self.L: (t + 1) * self.L] = self.A[t * self.L: (t + 1) * self.L, idxSS] * symbols[t]

            for m in range(self.M):
                YTempSymbols[:, m] = A * H[k, m]


            # --- Add the new matrix to the output signal
            Y += np.vstack((YTempPilots, YTempSymbols))


        return Y



    def receiver(self, Y):

        '''
        Function to recover the messages of the users from noisy observations
        Input:  The received signal, dimensions of Y, n x M
        Output: Probability of Detection and False Alarm
        '''


        # --- Save the received signal
        self.Y = Y.copy()

        # =========================================== Receiver  =========================================== #
        while True:

            # ======================== Pilot / Spreading Sequence Detector ======================== #
            self.idxSSHat = energyDetector(self, self.Y, self.K - self.count)


            # ======================== Channel estimation (Pilots) ======================== #
            HhatNew = channelEst(self)
    

            # ======================== Symbol estimation and Polar Code ======================== #
            userDecRx, notUserDecRx, symbolsHatHard, msgsHat2Part = decoder(self, HhatNew, self.idxSSHat)


            # ======================== NOPICE without Polar Decoder since is done above ======================== #
            if self.NOPICE:

                # --- Estimate the channel using P and Q
                HhatNew2 = channelEstWithErrors(self, symbolsHatHard)

                # --- Symbol Estimation and polar code
                userDecRx2, notUserDecRx2, symbolsHatHard2, msgsHat2Part2 = decoder(self, HhatNew2, self.idxSSHat)

                if userDecRx2.size >= userDecRx.size:
                    userDecRx = userDecRx2
                    notUserDecRx = notUserDecRx2
                    symbolsHatHard = symbolsHatHard2
                    msgsHat2Part = msgsHat2Part2


            # --- Add the new indices
            self.idxSSDec = np.append(self.idxSSDec, self.idxSSHat[userDecRx])


            # ======================== Exit Condition ======================== #
            # --- No new decoded user
            if userDecRx.size == 0:
                print('=== Done ===')
                DE, FA = checkPerformance(self)
                return DE, FA, self.count

            # ======================== Channel estimation (P + Q) ======================== #
            # --- Estimate the channel of the correct users
            # Use the received signal
            self.Y = Y.copy()
            HhatNewDec = channelEstWithDecUsers(self, Y, self.idxSSDec, symbolsHatHard[userDecRx])

            # ================================== SIC ================================== #
            # Only one user is decoded
            if userDecRx.size == 1:
                # Only one user left
                if msgsHat2Part.shape[0] == 1:
                    if not isIncluded(self, msgsHat2Part, self.idxSSHat[userDecRx]):
                        Hsub = np.squeeze(HhatNewDec.reshape(self.M, 1))

                        subInter(self, np.squeeze(symbolsHatHard), self.idxSSHat, Hsub)
                        saveUser(self, msgsHat2Part, self.idxSSHat[userDecRx])

                # More than one user left
                else:
                    if not isIncluded(self, msgsHat2Part[userDecRx, :], self.idxSSHat[userDecRx]):
                        Hsub = np.squeeze(HhatNewDec)
                        subInter(self, np.squeeze(symbolsHatHard[userDecRx, :]), self.idxSSHat[userDecRx], Hsub)
                        saveUser(self, msgsHat2Part[userDecRx, :], self.idxSSHat[userDecRx])

            # More than one user decode
            else:
                Hsub = HhatNewDec

                for g in range(userDecRx.size):
                    if not isIncluded(self, msgsHat2Part[userDecRx[g], :], self.idxSSHat[userDecRx[g]]):
                        subInter(self, symbolsHatHard[userDecRx[g]], self.idxSSHat[userDecRx[g]], Hsub[g, :])
                        saveUser(self, msgsHat2Part[userDecRx[g], :], self.idxSSHat[userDecRx[g]])


            # ======================== Find the performance ======================== #
            de, fa = checkPerformance(self)
            print('Number of Detections: ' + str(de))
            print('Number of False Alarms: ' + str(fa))
            print()


            
            # ======================== Exit Condition ======================== #
            if self.count == self.K:
                print('=== Done ===')
                DE, FA = checkPerformance(self)
                return DE, FA, self.count



# ============================================ Functions ============================================ #
# === General Functions
def QPSK(data):
    # Reshape data
    data = np.reshape(data, (2, -1))
    symbols = 1.0 - 2.0 * data

    # return (symbols[0, :] + 1j * symbols[1, :])
    return np.sqrt(0.5)*(symbols[0, :] + 1j * symbols[1, :])

def LMMSE(y,A,Qx,Qn):

    '''
        Function that implements the LMMSE Traditional and Compact Form estimator
        for the system y = Ax + n

        Input: y: received vector, A: measurement matrix, Qx: covariance of x, Qn: covariance of the noise
        Output: xHat: estimate of x, MSE: covariance matrix of the error
    '''

    r, c = A.shape

    # ========================= Traditional LMMSE ========================= #
    # https://en.wikipedia.org/wiki/Minimum_mean_square_error
    if r <= c:

        # Covariance Cov(X,Y)
        Qxy = np.dot(Qx, A.conj().T)

        # Covariance of Y
        Qy = np.dot(np.dot(A, Qx), A.conj().T) + Qn

        # Inverse of Covariance of Y
        QyInv = np.linalg.inv(Qy)

        # --- Filter
        F = np.dot(Qxy, QyInv)


    # ========================= Compact LMMSE ========================= #
    else:
        # --- Inverse of matrices
        QxInv = np.linalg.inv(Qx)

        QnInv = np.linalg.inv(Qn)

        # --- Second Term
        W2 = np.dot(A.conj().T, QnInv)

        # --- First term
        W1 = QxInv + np.dot(W2, A)
        W1Inv = np.linalg.inv(W1)

        # --- Filter
        F = np.dot(W1Inv, W2)


    # --- Estimates
    xHat = np.dot(F,y)

    return xHat

def isIncluded(self, second, idxSS):
    # --- Convert the decimal index of the spreading sequence to the binary string of length self.Bf
    first = dec2bin(np.hstack((idxSS, idxSS)), self.Bf)
    # --- Concatenate the two parts
    msgHat = np.append(first[0, :], second)

    # --- Check if we recovered this message
    for i in range(self.count):
        # --- Binary Addition
        binSum = sum((msgHat + self.msgsHat[i, :]) % 2)

        if binSum == 0:
            return 1
    return 0

def subInter(self, symbols, idxSS, h):

    # Define a temp Matrix and fill the matrix
    YTempPilots = np.zeros((self.nPilots, self.M), dtype=complex)
    YTempSymbols = np.zeros((self.nDataSlots * self.L, self.M), dtype=complex)

    # --- For Pilots
    for m in range(self.M):
        YTempPilots[:, m] = np.squeeze(self.P[:, idxSS]) * h[m]

    # --- For Symbols
    A = np.zeros((self.nDataSlots * self.L), dtype=complex)
    for t in range(self.nDataSlots):
        A[t * self.L: (t + 1) * self.L] = np.squeeze(self.A[t * self.L: (t + 1) * self.L, idxSS]) * symbols[t]

    for m in range(self.M):
        YTempSymbols[:, m] = A * h[m]

    # Subtract (SIC)
    self.Y -= np.vstack((YTempPilots, YTempSymbols))


def saveUser(self, msg2Part, idxSS):
    self.msgsHat[self.count, :] = np.concatenate((np.squeeze(dec2bin(np.array([idxSS]), self.Bf)), np.squeeze(msg2Part)), 0)
    self.count += 1

def checkPerformance(self):
    numDE, numFA = 0, 0
    for i in range(self.count):
        flag = 0
        for k in range(self.K):
            binSum = sum((self.msgs[k, :] + self.msgsHat[i, :]) % 2)

            if binSum == 0:
                flag = 1
                break
        if flag == 1:
            numDE += 1
        else:
            numFA += 1

    return numDE, numFA

# === Energy Detector
def energyDetector(self, y, K):

    # --- Energy Per Antenna
    energy = np.linalg.norm(np.dot(self.P.conj().T, y[0:self.nPilots, :]), axis=1) ** 2

    pivot = self.nPilots
    for t in range(self.nDataSlots):
        energy += np.linalg.norm(np.dot(self.A[t * self.L: (t + 1) * self.L, :].conj().T,
                                        y[pivot + t * self.L: pivot + (t + 1) * self.L, :]), axis=1) ** 2

    return np.argpartition(energy, -K)[-K:]

# ==== Functions For Channel Estimation
def channelEst(self):
    return LMMSE(self.Y[0:self.nPilots, :], self.P[:, self.idxSSHat], np.eye(len(self.idxSSHat)), np.eye(self.nPilots) * self.sigma2)

def channelEstWithErrors(self, symbolsHatHard):
    K = symbolsHatHard.shape[0]

    # -- Create A
    A = np.zeros((self.nChanlUses, K), dtype=complex)
    for k in range(K):
        Atemp = np.zeros((self.nDataSlots * self.L), dtype=complex)
        for t in range(self.nDataSlots):
            Atemp[t * self.L: (t + 1) * self.L] = self.A[t * self.L: (t + 1) * self.L, self.idxSSHat[k]] * symbolsHatHard[k, t]

        A[:, k] = np.hstack((self.P[:, self.idxSSHat[k]], Atemp))

    return LMMSE(self.Y, A, np.eye(K), np.eye(self.nChanlUses) * self.sigma2)

def channelEstWithDecUsers(self, Y, decUsersSS, symbolsHatHard):


    for i in range(self.count - 1, -1, -1):
        symbolsHatHard = np.vstack((self.symbolsHat[i, :], symbolsHatHard))

    K = decUsersSS.size

    # -- Create A
    A = np.zeros((self.nChanlUses, K), dtype=complex)
    for k in range(K):
        Atemp = np.zeros((self.nDataSlots * self.L), dtype=complex)
        for t in range(self.nDataSlots):
            if K > 1:
                Atemp[t * self.L: (t + 1) * self.L] = self.A[t * self.L: (t + 1) * self.L, decUsersSS[k]] * symbolsHatHard[k, t]
            else:
                Atemp[t * self.L: (t + 1) * self.L] = np.squeeze(self.A[t * self.L: (t + 1) * self.L, decUsersSS] * symbolsHatHard[t])
        if K > 1:
            A[:, k] = np.hstack((self.P[:, decUsersSS[k]], Atemp))
        else:
            A[:, k] = np.hstack((self.P[:, decUsersSS], Atemp))

    Hhat = LMMSE(Y, A, np.eye(K), np.eye(self.nChanlUses) * self.sigma2)

    for i in range(self.count):
        subInter(self, self.symbolsHat[i, :], decUsersSS[i], Hhat[i, :])

    return Hhat[self.count::, :]

# ==== Functions For Symbol Estimation
def symbolsEst(Y, H, A, Qx, Qn, nSlots, L):

    '''
        Function that implements the symbols estimation for spread-based MIMO system

        X: diagonal matrix contains the symbol for each user, H: channel matrix, N: noise

        Input: Y: received vector, H: channel matrix,
               A: spreading sequence master set (for all slots, only active) dim(A) = (L x nSlots) x totalNumber of Spreading Sequence,
               Qx: covariance of x, Qn: covariance of the noise
               nSlots: number of slots which is equal to the number of symbols
               L: Length of the spreading sequence
        Output: xHat: estimate of x, MSE: covariance matrix of the error
    '''


    K = H.shape[0]

    symbolsHat = np.zeros((K, nSlots), dtype=complex)

    # --- For all Symbols
    for t in range(nSlots):
        symbolsHat[:, t]  = symbolEstSubRoutine(Y[t * L: (t + 1) * L, :], H, A[t * L:(t + 1) * L, :], Qx, Qn)

    return symbolsHat

def symbolEstSubRoutine(Y,H,S,Qx,Qn):

    '''
        Function that implements the symbol estimation for spread-based MIMO system Y = SXH + N
        where Y: received matrix, S: spreading sequence matrix (only active columns),
        X: diagonal matrix contains the symbol for each user, H: channel matrix, N: noise

        Input: Y: received vector, H: channel matrix, S: spreading sequence matrix ,
               Qx: covariance of x, Qn: covariance of the noise
        Output: xHat: estimate of x, MSE: covariance matrix of the error
    '''

    K, M = H.shape # K: number of users, M: number of antennas
    L = S.shape[0] # L: length of the spreading sequence

    # --- First Step:
    # Convert the system from Y = SXH + N, to y = Ax + n, where A contains the channel and sequence

    A = np.zeros((L * M, K), dtype=complex)

    for m in range(M):
        if K == 1:
            A[m * L:L * (m + 1), :] = S * H[:,m]
        else:
            # --- Diagonalize H
            A[m * L:L * (m + 1), :] = np.dot(S, np.diag(H[:, m]))

    # --- Second Step:

    # Flat Y
    y = np.ndarray.flatten(Y.T)

    # Estimate the symbols
    return LMMSE(y,A,Qx,Qn)

# ==== Decoder
def decoder(self, H, idxSSHat):

    K = idxSSHat.size
    symbolsHatHard = np.zeros((K, self.nDataSlots), dtype=complex)


     # ==================================== Symbol Estimation Decoder ============================================== #
    symbolsHat = symbolsEst(self.Y[self.nPilots::, :], H, self.A[:, idxSSHat], np.eye(K),
                            np.eye(self.L * self.M) * self.sigma2, self.nDataSlots, self.L)

    # ==================================== Channel Decoder ============================================== #
    userDecRx = np.array([], dtype=int)
    notUserDecRx = np.array([], dtype=int)
    msgsHat = np.zeros((K, self.Bs), dtype=int)
    c = 0
    for s in range(symbolsHat.shape[0]):
        # Form the codeword
        cwordHatSoft = np.concatenate((np.real(symbolsHat[s, :]), np.imag(symbolsHat[s, :])), 0)

        # Interleaver
        cwordHatSoftInt = np.zeros(self.nc)
        cwordHatSoftInt[self.interleaver[:, self.idxSSHat[s]]] = cwordHatSoft

        # Call polar decoder
        cwordHatHard, isDecoded, msgHat = polarDecoder(self, np.sqrt(2)*cwordHatSoftInt, self.idxSSHat[s])

        if isDecoded == 1 and sum(abs(((cwordHatSoftInt < 0) * 1 - cwordHatHard)) % 2) > self.nc/2:
            isDecoded = 0

        symbolsHatHard[s, :] = QPSK(cwordHatHard[self.interleaver[:, self.idxSSHat[s]]])
        msgsHat[s, :] = msgHat



        if isDecoded:
            userDecRx = np.append(userDecRx, s)
        else:
            notUserDecRx = np.append(notUserDecRx, s)


    return userDecRx, notUserDecRx, symbolsHatHard, msgsHat

def polarDecoder(self, bitsHat, idxSSHat):
    # ============ Polar decoder ============ #
    msgCRCHat, PML = self.polar.listDecoder(bitsHat, self.frozenValues[:, idxSSHat], self.nL)

    # ============ Check CRC ============ #
    # --- Initialization
    thres, flag = np.Inf, -1
    isDecoded = 0

    # --- Check the CRC constraint for all message in the list
    for l in range(self.nL):
        check = crcDecoder(msgCRCHat[l, :], self.divisor)
        if check:
            # --- Check if its PML is larger than the current PML
            if PML[l] < thres:
                flag = l
                thres = PML[l]
                isDecoded = 1

    if thres == np.Inf:
        # --- Return the message with the minimum PML
        flag = np.squeeze(np.where(PML == PML.min()))

    # --- Encode the estimated message
    codewordHat, _ = self.polar.encoder(msgCRCHat[flag, :], self.frozenValues[:, idxSSHat], -1)

    return codewordHat, isDecoded, msgCRCHat[flag, 0:self.Bs]


