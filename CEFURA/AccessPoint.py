import numpy as np
from utilities import LMMSE, scatterPlot, demQPSK
import matplotlib.pyplot as plt


class APs:
    def __init__(self, nUEs, nAPs, nAnts, B, Bf, L, nc, nL, nPilots, sigma2, UEs, nRec, net):
        ''' Parameters '''
        self.nUEs = nUEs  # Number of Users
        self.nAPs = nAPs  # Number of APs
        self.nAnts = nAnts # Number of antennas per AP
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
        self.UEs = UEs
        self.nRec = int(nRec)
        self.save = 0
        self.net = net
        self.Pt = self.UEs.Pt

        ''' For polar code '''
        self.divisor = self.UEs.divisor

        self.lCRC = len(self.divisor)  # Number of CRC bits
        self.msgLen = self.Bs + self.lCRC  # Length of the input to the encoder
        self.frozenValues = self.UEs.frozenValues
        self.idx2UE = self.UEs.idx2UE
        self.UE2idx = self.UEs.UE2idx

        ''' Generate matrices '''
        # Pilots
        self.P = self.UEs.P

        # Spreading sequence master set
        self.A = self.UEs.A

        ''' To store information '''
        self.idxSSHat = np.zeros((self.nAPs, self.nRec), dtype=int) - 1
        self.symbolsHat = np.zeros((self.nRec, self.nAPs, self.nQPSKSymbols), dtype=complex)
        self.recPerAP = np.zeros(self.nAPs, dtype=int)
        self.Hhat = np.zeros((self.nRec, self.nAPs, self.nAnts), dtype=complex)



    def receiver(self, Y):

        # ====================== Prepare the available data for processing ====================== #
        # Pilot Signal
        self.yPilots = Y[0:self.nPilots, :]

        # Data Signal
        self.yData = Y[self.nPilots::, :]
        yData = np.swapaxes(np.reshape(Y[self.nPilots::, :], (self.nQPSKSymbols, self.L, self.nAPs, self.nAnts)), 0, 1)

        # ====================== Sequence Detector and Channel Estimation ====================== #
        self.OMP(self.yPilots)

        # ================================== Symbol Estimation ================================== #
        if self.nAnts == 1:
            self.Hhat = np.squeeze(self.Hhat)
            yData = np.squeeze(yData)

        # --- Symbol Estimation/ For all APs
        for ap in range(self.nAPs):

            if self.nAnts == 1:
                self.symbolsHat[:, ap, :] = symbolEst(yData[:,:,ap], self.A[:, self.idxSSHat[ap, :]], np.diag(self.Hhat[:,ap]), self.sigma2)
            else:
                # --- Combine the signals from different antennas
                y, B = processData(yData[:, :, ap, :], self.A[:, self.idxSSHat[ap, :]], self.Hhat[:, ap, :], self.L, self.nAnts, self.nRec)

                self.symbolsHat[:, ap, :] = LMMSE(y, B, np.eye(self.nRec), self.sigma2 * np.eye(self.L * self.nAnts))
                
        return self.symbolsHat, self.idxSSHat


    def OMP(self, yPilots):


        # ====================== Prepare the available data for processing ====================== #
        yPilots3D = np.reshape(yPilots, (self.nPilots, self.nAPs, self.nAnts)).copy()
        yResidual = yPilots.copy()

        # === Recover at max nRec users
        for _ in range(self.nRec):
            # --- Step 1: Energy Detector
            idxSSHat = energyDetector(self, yResidual)

            # --- Step 2: Channel Estimation
            for ap in range(self.nAPs):

                # 1) Find the active pilot sequences
                self.idxSSHat[ap, self.recPerAP[ap]] = idxSSHat[ap]
                self.recPerAP[ap] += 1


                # 2) Channel Estimation
                self.Hhat[0:self.recPerAP[ap], ap, :] = channelEstML(yPilots3D[:, ap, :], self.P[:, self.idxSSHat[ap, 0:self.recPerAP[ap]]])

                # 3) Successive Interference Cancellation
                yResidual[:, self.nAnts * ap:self.nAnts * (ap + 1)] = yPilots[:, self.nAnts * ap:self.nAnts * (ap + 1)] \
                                                                      - np.dot(self.P[:, self.idxSSHat[ap,  0:self.recPerAP[ap]]], self.Hhat[0:self.recPerAP[ap], ap, :])


        return checkEnergyDetector(self)



# ============================================ Functions ============================================ #
def energyDetector(self, y):
    # Energy Per Antenna Per AP
    energy = abs(np.dot(self.P.conj().T/np.sqrt(self.Pt), y)) ** 2

    # Combine only the energy from the same AP
    energy = np.sum(np.reshape(energy, (energy.shape[0], self.nAPs, self.nAnts)), axis=2)

    idx = np.argmax(energy, 0)
    for ap in range(self.nAPs):
        while idx[ap] in self.idxSSHat[ap,:]:
            energy[idx, ap] = -1
            idx[ap] = np.argmax(energy[:,ap], 0)

    return idx

def channelEstML(y,P):
    PT =  P.conj().T
    PPinv = np.linalg.inv(PT @ P)
    Py = np.dot(PT, y)

    return np.dot(PPinv,Py)

def symbolEst(y, A, H, sigma2):
    L, nUE = A.shape
    B = A @ H
    return LMMSE(y, B, np.eye(nUE), sigma2 * np.eye(L))


def processData(Y, A, H, L, nAnts, nRec):
    # 1) reshape y
    yReshape = np.reshape( np.swapaxes(Y, 1,2), (L * nAnts, -1), order='F')

    # 2) Create B
    B = np.zeros((yReshape.shape[0], nRec), dtype=complex)

    for i in range(nAnts):
        B[i * L:(i + 1) * L, :] = A @ np.diag(H[:, i])

    return yReshape, B

