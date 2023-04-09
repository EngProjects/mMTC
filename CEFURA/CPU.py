import numpy as np
from PolarCode import PolarCode
from utilities import dec2bin, crcDecoder

class CPU:
    def __init__(self, nUEs, nAPs, B, Bf, nc, nL, sigma2):
        ''' Parameters '''
        self.nUEs = nUEs  # Number of Users
        self.nAPs = nAPs  # Number of APs
        self.B = B
        self.Bf = Bf  # number of bits of the first part of the message
        self.Bs = B - Bf  # number of bits of the second part of the message
        self.nc = nc  # length of code
        self.nL = nL  # List size
        self.nQPSKSymbols = int(nc / 2)  # Number of QPSK symbols
        self.sigma2 = sigma2

        ''' For polar code '''
        # Polynomial for CRC coding
        self.divisor = np.array([1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1], dtype=int)

        self.lCRC = len(self.divisor)  # Number of CRC bits
        self.msgLen = self.Bs + self.lCRC  # Length of the input to the encoder

        self.lCRC = len(self.divisor)  # Number of CRC bits
        self.msgLen = self.Bs + self.lCRC  # Length of the input to the encoder


    def combine(self, symbolsHat, idxSSHat, frozenvalues, messages, idx2UE, codewords):

        self.count = 0


        uniqIdxTemp, frequency = np.unique(np.ndarray.flatten(idxSSHat), return_counts=True)
        sorted_indexes = np.argsort(frequency)[::-1]
        uniqIdx = uniqIdxTemp[sorted_indexes]
        self.msgsHat = np.zeros((len(uniqIdx), self.B))
        # Create a polar code object
        polar = PolarCode(self.nc, self.msgLen, 1)


        freq = np.zeros(len(uniqIdx))
        # --- Use the index of the pilot to combine the signals
        for ue, actIdx in enumerate(uniqIdx):

            # Find which of the APs have symbols for this user (Assuming no collisions)
            apIdx, ueIdx = np.where(actIdx == idxSSHat)
            freq[ue] = len(apIdx)

            # Combine the symbols
            currSymb = sum(symbolsHat[ueIdx, apIdx, :])


            # Call Polar Decoder
            msgHat, isDecoded = channDecoder(currSymb, polar, frozenvalues[:, actIdx], self.nL, self.divisor)

            if isDecoded:

                firsPart = dec2bin(np.array([actIdx]), self.Bf)[0]
                self.msgsHat[self.count, 0:self.Bf] = firsPart
                self.msgsHat[self.count, self.Bf::] = msgHat[0:self.Bs]
                self.count += 1
            if self.count >= self.nUEs + int(0.05*self.nUEs):
                break



        nDE, nFA = checkPerformance(self, messages)


        return nDE, nFA, self.count

def channDecoder(symbolsHat, polar, frozenValues, nL, divisor):

        # Demodulate the QPSK symbols
        cwordHatSoft = np.concatenate((np.real(symbolsHat), np.imag(symbolsHat)), 0)

        # ============ Polar decoder ============ #
        msgCRCHat, PML = polar.listDecoder(np.sqrt(2) * cwordHatSoft, frozenValues, nL)

        # ============ Check CRC ============ #
        # --- Initialization
        thres, flag = np.Inf, -1

        # --- Check the CRC constraint for all message in the list
        for l in range(nL):
            check = crcDecoder(msgCRCHat[l, :], divisor)
            if check:
                # --- Check if its PML is larger than the current PML
                if PML[l] < thres:
                    flag = l
                    thres = PML[l]

        # --- Encode the estimated message
        if thres != np.Inf:
            msg2Hat = msgCRCHat[flag, :]
            isDecoded = 1
        else:
            msg2Hat = 0
            isDecoded = 0


        return msg2Hat, isDecoded


def checkPerformance(self, msgs):
    nDE, nFA = 0, 0
    for i in range(self.count):
        flag = 0
        for k in range(self.nUEs):
            binSum = sum((msgs[k, :] + self.msgsHat[i, :]) % 2)

            if binSum == 0:
                flag = 1
                break
        if flag == 1:
            nDE += 1
        else:
            nFA += 1

    return nDE, nFA
