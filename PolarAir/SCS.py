import numpy as np
from PolarCode import PolarCode
from helper import bin2dec, dec2bin, crcEncoder, crcDecoder
import time


class SCS():
    def __init__(self, A, nc, Bf, Bs, divisor, frozenValues, K, nL, nPar):
        self.A = A  # spreading sequence master set
        self.nc = nc  # length of code
        self.Bf = Bf  # number of bits of the first part of the message
        self.Bs = Bs  # number of bits of the second part of the message
        self.divisor = divisor  # polynomial
        self.L = A.shape[0]  # Length of spreading sequence
        self.J = A.shape[1]  # Number of spreading sequence
        self.frozenValues = frozenValues  # Frozen values for each user
        self.K = K  # Number of active sequence
        self.nL = nL  # List size
        self.nPar = nPar
        self.support = np.zeros(nPar, dtype=int)
        self.values = np.zeros(K, dtype=int)
        self.idxMs = np.zeros(K, dtype=int)  # To save the decimal value of the first part of the message
        self.idxSS = np.zeros(K, dtype=int)  # To save the indices of the spreading sequence
        self.idxMsCRC = np.zeros(K, dtype=int) - 1  # To save the decimal value the message + CRC
        self.msgLen = len(divisor) + Bs  # Length of the input to the encoder
        self.checkSS = np.zeros(K + 1, dtype=int)  # To check which of the spreading sequences are detected
        self.Y = np.zeros((self.L, self.nc))  # To save the receive signal
        self.supportHat = np.zeros(nPar, dtype=int)
        self.supportHatDec = np.zeros(K, dtype=int) - 1  # To save the support
        self.valuesHat = np.zeros(K)  # To save the values
        self.Phi = np.zeros((self.L * self.nc, 1))  # To save the active columns of the measument matrix
        self.countBef = 0  # Count the number of recovered msgs in the brevious rounds
        self.xHat = np.zeros(self.L * self.nc)  # To save the recovered sparse vector
        self.codewordIdx = np.zeros(K)

    def saveSupp(self, idx):
        self.support[idx] = 1

    def sampling(self, msgBin, power, polar):
        """ Function to sampling the sparse vector
            Inputs:
                1. msgBin: Binary 2D array which contains the active indices in binary form
                           Dimensions of msgBin K x B
                2. power: The corresponding amplitudes
                3. polar: Object to use during the polar encoding
            Outputs:
                1. x: Measurement vector before the noise get added to it
                      Dimensions of msgBin L x nc
        """

        # ===================== Initialization ===================== #
        x = np.zeros((self.L, self.nc))  # To store the measurement vector

        debug = 1

        # --- For all active indices
        for k in range(self.K):

            # --- Break the message into to 2 parts
            mf = msgBin[k, 0:self.Bf]  # First part
            ms = msgBin[k, self.Bf::]  # Second part

            # --- Find the decimal representation of ms
            self.idxSS[k] = bin2dec(ms)

            # --- Encode mf
            msgCRC = crcEncoder(mf, self.divisor)

            # Call polar encoder
            codewrd, _ = polar.encoder(msgCRC, self.frozenValues[:, self.idxSS[k]])

            # BPSK modulation
            codewrd = 1 - 2 * codewrd

            # Define a temp Matrix and fill the matrix
            xTemp = np.zeros((self.L, self.nc))

            # --- For all modulated bits
            for i in range(self.nc):
                # Spread the modulated bit
                xTemp[:, i] = self.A[:, self.idxSS[k], i] * codewrd[i]

            # Add the new column of the measurement matrix to the sketch
            x += power[k] * xTemp

        if debug:
            self.checkSS = -np.ones(len(np.unique(self.idxSS) + 1), dtype=int)
            # print(f'Unique Spre-Seq = {len(np.unique(self.idxSS))}')

        return x

    def recover(self, y, polar):

        """ Function to recover the sparse vector
            Inputs:
                1. y: Measurements, dimensions of msgBin K x B
                2. polar: Object to use during the polar decoding
            Outputs:
                1. x: Measurement vector before the noise get added to it
                      Dimensions of msgBin L x nc
        """

        # --- Save the receive signal
        self.Y = y
        nIter = 0
        self.count = 0  # Count the number of recovered msgs in this round
        # --- Start the decoding
        while 1:

            # --- Count the number of iterations
            nIter += 1
            
            # ================== Energy Detector ====================== #
            idxSSHat, outMF = spreSeqDetector(self, self.Y, self.K - self.count)


            
            # ================== Polar Decoder ====================== #
            for k in range(len(idxSSHat)):
                # --- Call Polar decoder
                msgHat_p, PML_p = polar.listDecoder(outMF[k, :], self.frozenValues[:, idxSSHat[k]], self.nL)
                msgHat_n, PML_n = polar.listDecoder(-1 * outMF[k, :], self.frozenValues[:, idxSSHat[k]], self.nL)

                # --- Check CRC
                thres_p, thres_n = np.Inf, np.Inf
                flag_p, flag_n = -1, -1
 
                # start = time.time()
                # --- For positive
                for l in range(self.nL):

                    check = crcDecoder(msgHat_p[l, :], self.divisor)
                    if check:
                        # --- Check if its PML is larger than the current PML
                        if PML_p[l] < thres_p:
                            flag_p = l
                            thres_p = PML_p[l]


                # --- For negative
                for l in range(self.nL):

                    check = crcDecoder(msgHat_n[l, :], self.divisor)
                    if check:
                        # --- Check if its PML is larger than the current PML
                        if PML_n[l] < thres_n:
                            flag_n = l
                            thres_n = PML_n[l]

                # --- Check the cases
                # --- Positive is correct
                if thres_p < np.Inf and thres_n == np.Inf:
                    msgHat = msgHat_p[flag_p, :]

                # --- Negative is correct
                elif thres_n < np.Inf and thres_p == np.Inf:
                    msgHat = msgHat_n[flag_n, :]
                    outMF[k, :] = -1 * outMF[k, :]

                # --- Both are correct
                elif thres_n < np.Inf and thres_p < np.Inf:
                    # --- Check the PML
                    if thres_n < thres_p:
                        msgHat = msgHat_n[flag_n, :]
                        outMF[k, :] = -1 * outMF[k, :]
                    else:
                        msgHat = msgHat_p[flag_p, :]

                        # --- None of them
                elif thres_p == np.Inf and thres_n == np.Inf:
                    # --- Go to next msg
                    continue


                # ================== Polar Encoder ====================== #
                hard = 1 * (outMF[k, :] < 0)
                codeHat, _ = polar.encoder(msgHat, self.frozenValues[:, idxSSHat[k]])
                if sum((hard + codeHat) % 2) < 20:

                    # ================== Add the new Column ====================== #
                    if isInSupport(self, msgHat, idxSSHat[k]):
                        # --- If it is a new index
                        # --- At this column to the temp measument matrix
                        phiContr(self, msgHat, idxSSHat[k], polar)
                else:
                    continue


            # ========================== Estimate values  ============================== #
            valuesHat = estValues(self)

            # ========================== SIC ============================== #
            sIC(self, valuesHat)

            # =============================== Exit conditions ========================= #
            if self.count >= self.K: # Recover more or equal to K
                return self.count, constrX(self), self.supportHatDec, nIter

            # No improvement in this round
            elif self.count == self.countBef:
                return self.count, constrX(self), self.supportHatDec, nIter

            # The number of iterations are more or equal to 20
            elif nIter >= 20:
                return self.count, constrX(self), self.supportHatDec, nIter
            else:
                self.countBef = self.count


def spreSeqDetector(self, y, K):
    energy = np.zeros(self.J)  # Store the Energy
    outMF = np.zeros((self.J, self.nc))  # Store the Estimates
    # --- For each spreading sequence
    for t in range(self.nc):
        outMF[:, t] = np.dot(self.A[:, :, t].T, y[:, t])  # Matched Filter
        energy += abs(outMF[:, t]) ** 2  # Compute the energy (Multiple Observations)

    # Active indices
    activeIndices = np.argpartition(energy, -K)[-K:]

    return activeIndices, outMF[activeIndices, :]


def constrX(self):
    """ Function to construct the sparse vector
    """
    # --- Initialize a vector with dimensions nPar
    gradient = np.zeros(self.nPar)

    # --- Save the values to the corresponding active indices
    gradient[self.supportHatDec[0:self.count]] = self.valuesHat[0:self.count]

    return gradient


def sIC(self, valuesHat):
    """ Function to perform the Successive Interference Cancelation
        Input:
            1. New Estimated values
        Output:
            1. New residual

    """

    # --- Take the column of Phi
    inChan = self.Phi[:, 1::]

    # --- Dimensions m x self.count
    X = inChan * np.squeeze(valuesHat)

    # --- Dimensions m x 1
    Y = np.reshape(self.Y.T, (-1, 1))

    # --- Compute the energy of Y
    # --- SIC
    Y = np.squeeze(Y) - np.sum(X, axis=1)

    # --- Save the residual
    self.Y = np.reshape(Y, (-1, self.L)).T


def estValues(self):
    """ Function to perform the Least Square Estimator

        Output:
            1. xHat: New values

    """

    # --- Apply least squares
    phiPhiT = np.matmul(self.Phi[:, 1::].T, self.Phi[:, 1::])
    # phiPhiT = phiPhiT+ np.random.normal(loc =0,scale = 0.0001, size= (phiPhiT.shape[0],phiPhiT.shape[0]))

    invPhi = np.linalg.inv(phiPhiT)

    # --- Change the size of the residual from L x nc to Lnc x 1 to perfom the LS estimator
    yRe = np.reshape(self.Y.T, (-1, 1))

    # --- LS estimates
    xHat = np.dot(np.matmul(invPhi, self.Phi[:, 1::].T), yRe)

    # --- Update/ Correct the previous values
    self.valuesHat[0:self.count] = self.valuesHat[0:self.count] + np.squeeze(xHat)

    return xHat


def phiContr(self, msgCRC, idx, polar):
    # Call polar encoder
    codewrd, _ = polar.encoder(msgCRC, self.frozenValues[:, idx])

    # BPSK modulation
    codewrd = 1 - 2 * codewrd

    # Form the column
    xTemp = np.zeros((self.L, self.nc))
    for i in range(self.nc):
        xTemp[:, i] = self.A[:, idx, i] * codewrd[i]

    # Reshape
    xRe = np.reshape(xTemp.T, (-1, 1))

    # Add column
    self.Phi = np.hstack((self.Phi, xRe))


def isInSupport(self, msgHat, idxSSHat):
    """ Function to check if the recovered index is new or not
        Input:
            1. msgHat: recovered codeword + CRC
            2. idxSSHat: recovered index

        Output:
            1. Binary 1 or 0 is the index is new or not respectively

    """

    # === Find the binary representation  of the active index
    # --- First part
    mfHat = msgHat[0:self.Bf]
    # --- Second part
    msHat = dec2bin(np.hstack((idxSSHat, idxSSHat)), self.Bs)
    # --- Combine
    mHat = np.hstack((mfHat, msHat[0, :]))

    # === Find the decimal representation  of the active index
    idxDec = bin2dec(mHat)

    # --- If the index is larger than the number of parameters, discard
    if idxDec >= self.nPar:
        return 0
    # --- If the index is already recovered, discard
    elif self.supportHat[idxDec] > 1:
        return 0
    # --- If the index is new, save it
    elif self.supportHat[idxDec] == 0:
        # --- Check that this index is recovered
        # self.supportHat[idxDec] = self.supportHat[idxDec] + 1
        self.supportHat[idxDec] = 1
        # --- Save the index
        self.supportHatDec[self.count] = idxDec
        # --- Increase the number of the recovered indices
        self.count += 1
        return 1


# ---- Define functions
def changePar(N, K, J, msgLen, L, nc):
    if nc < 64:
        # increase the length of the code
        nc = nc * 2
    else:
        # increase the length of the spreading sequence
        L = L + 100

    # Generate the Master set of spreading sequence
    A = (1 - 2 * np.round(np.random.randint(low=0, high=2, size=(L, J, nc)))) / np.sqrt(N)

    # Pick the frozen Values randomly
    frozenValues = np.round(np.random.randint(low=0, high=2, size=(nc - msgLen, J)))

    # Create a polarcode object
    polar = PolarCode(nc, msgLen, K)

    # Total number of measuments
    s = L * nc

    return s, L, nc, A, frozenValues, polar


def initPar(N, K, J, msgLen, Ltemp, ncTemp):
    # Length of spreading sequence
    L = Ltemp

    # Length of code
    nc = ncTemp

    # Generate the Master set of spreading sequence
    A = (1 - 2 * np.round(np.random.randint(low=0, high=2, size=(L, J, nc)))) / np.sqrt(N)

    #  Pick the frozen Values randomly
    frozenValues = np.round(np.random.randint(low=0, high=2, size=(nc - msgLen, J)))

    # Create a polarcode object
    polar = PolarCode(nc, msgLen, K)

    # Total number of measuments
    s = L * nc

    return s, L, nc, A, frozenValues, polar



"""
Function to check that everything works well

"""


def checkAccSs(self, idxSSHat):
    count = self.checkSS[0] + 1
    for s in idxSSHat:
        if s in self.supportHat:
            continue
        elif s in np.unique(self.idxSS):
            self.checkSS[count] = s
            count += 1

    self.checkSS[0] = count - 1


def checkAccMsCRC(self, msgHat):
    idxMsgHat = bin2dec(msgHat)
    if idxMsgHat in self.idxMsCRC:
        return 1
    else:
        return 0


def saveX(self):
    for k in range(self.K):
        self.support[k] = bin2dec(self.msgBin[k, :])


def checkSupp(self):
    FA, D = 0, 0
    # --- Find the nonzero values
    non = np.nonzero(self.supportHat)
    for s in non[0]:
        if self.support[s] == 1:
            D += 1
        else:
            FA += 1


