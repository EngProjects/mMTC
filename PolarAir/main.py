# -*- coding: utf-8 -*-
"""
Program to simulate the training of the CIFAR-10 dataset using 8 workers, who sparsify the gradient
K = 270, use the polar scheme to compress further the sparse gradient and use adaptive increasing of the
number of measurements
"""

# ---- Import packages
import torch
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torchvision.datasets as dset
from torch.utils.data import sampler
import numpy as np
import time

# ---- Import Files
from SCS import SCS, initPar, changePar
from helper import dec2bin
from Parties import Worker, Server, Experimentalist
from PolarCode import PolarCode
from Models import resnet20

# ---- Settings
np.set_printoptions(precision=4)
np.set_printoptions(linewidth=100000)

print('#--------- The simulation has started ---------#')

# =============================================== START the Simulation =============================================== #
# =========================================== Initialization (ML) =========================================== #
W = 8  # Number of workers
BZ = 256  # Batch Size
C = 24  # Total number of communication rounds per epoch
dataPerComm = BZ * W  # Total available data for one communication round
numTrainSamples = dataPerComm * C

# --- CNN
model = resnet20()  # Create an CNN
nPar = 269722  # Total number of parameters in ResNet20
K = 270  # 270 # Number of active parameters

# --- Optimizer
lr = 0.01  # Learning Rate
optimizer = optim.Adam(model.parameters(), lr=lr)  # Initialize Adam optimizer

# ============ Dataset ============ #
transform = T.Compose([T.Pad(4), T.RandomHorizontalFlip(), T.RandomCrop(32), T.ToTensor()])

# --- Data for Training
cifar10Train = dset.CIFAR10('dataset/', train=True, download=True, transform=transform)
# It takes a data set and returns batches of images and corresponding labels
loaderTrain = DataLoader(cifar10Train, batch_size=dataPerComm,
                         sampler=sampler.SubsetRandomSampler(range(numTrainSamples)))

# --- Data for Test
#cifar10Test = dset.CIFAR10('dataset/', train=False, download=True, transform=transform)
cifar10Test = dset.CIFAR10('dataset/', train=False, download=True, transform=T.ToTensor())
# It takes a data set and returns batches of images and corresponding labels
loaderTest = DataLoader(cifar10Test, batch_size=10000)

for (x, y) in loaderTest:
    xTest = x
    yTest = y

# =========================================== Initialization (CS) =========================================== #
# Length of the sparse vector
N = 2 ** 19
# Length of the message, Length of the First and Second part of the message
B, Bf = int(np.log2(N)), 10
Bs = B - Bf
# Number of spreading sequence
J = 2 ** Bs
# Length of the polar list and CRC divisor
nL, divisor = 16, np.array([1, 1, 0, 1, 0, 0, 1, 1], dtype=int)
lCRC = len(divisor)
# Length of the message before the polar encoder
msgLen = Bf + lCRC

# Initialize parameters
s, L, nc, A, frozenValues, polar = initPar(N, K, J, msgLen, 400, 32)

# USE Polar Scheme, USE power allocation, SAVE data
POLAR_SCHEME = 1
NOISE = 1
SAVE = 1

# Power and variance of the noise
P = 1000
sigma = 1

# ============================================ Activate GPU =============================================== #
dtype = torch.float32
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('using device:', device)

# ============================================ Initialize Workers =============================================== #
# Number of Workers
workers = []

for w in range(W):
    workers.append(Worker(K, nPar, device, dtype))

# Server
server = Server(K, nPar, device, dtype, model, optimizer)

# ========================================== Training ===================================================== #
epochs = 300
# Count the number of training iterations
countTr = 0
# Create an Experimentalist to save the results
expert = Experimentalist(K, int(10 * np.log10(P)), W, nPar, C, epochs, POLAR_SCHEME, NOISE)

# --- Total Time of the simulation
startS = time.time()
print('Start Simulation')

# ================================================= Epoch ================================================= #
# --- For all epochs
for i in range(epochs):

    # === Initialization
    startE = time.time()
    c = 0  # Counter to count the number of communication rounds
    print("Epoch " + str(i))

    # Check if the average of the number of the recovered indices was less than the half K
    if i > 0 and POLAR_SCHEME:
        # Compute the average
        aveIdx = sum(expert.numIdxHat[:, i - 1]) / float(C)

        if aveIdx < K / 2:
            # Increase the number of measurements
            s, L, nc, A, frozenValues, polar = changePar(N, K, J, msgLen, L, nc)

    # --- For all available Training Data
    for (x, y) in loaderTrain:
        # ================================================= Communication Round ================================================= #
        w = 0  # counter for the workers
        originalGrad = np.zeros(nPar)
        if not POLAR_SCHEME:
            s = nPar
            gradient = np.zeros(nPar)
        else:
            s = L * nc
            gradient = np.zeros((L, nc))
        if NOISE:
            alphas = 0
            mu = 0

        table = np.zeros(nPar)  # to count the number of times an index is active

        # ================================================= Training ================================================= #
        # --- For all workers
        for w in range(W):

            # Returns the dense gradient and the active indices
            meas, idx = workers[w].train(x[w * BZ:(w + 1) * BZ, :, :, :], y[w * BZ:(w + 1) * BZ], model, optimizer)

            # Compute the original gradient
            originalGrad = originalGrad + meas

            # Count the number of times an index is active
            table[idx] = table[idx] + 1

            # === Compress
            if POLAR_SCHEME:
                # --- Convert the messages to binary strings
                msgBin = dec2bin(idx, B)  # MSB-LSB
                # --- Create a scheme to compressed the gradient
                scheme = SCS(A, nc, Bf, Bs, divisor, frozenValues, K, nL, nPar)
                # --- Sample the sparse vector
                meas = scheme.sampling(msgBin, meas[idx], polar)

            # === If noise, scale the input to the channel
            if NOISE:
                # --- Transmit the whole gradient with power allocation
                # Subtruct mean
                sumG = np.sum(meas) / s
                # gradZero mean
                gradZeroMean = meas - sumG
                # power allocation
                gradSq = np.linalg.norm(np.ndarray.flatten(meas)) ** 2
                alpha = np.sqrt((P * (s + 2)) / (gradSq - (s - 1) * (sumG ** 2) + 1))
                alphas = alphas + alpha
                mu = mu + sumG * alpha
                # Scale
                meas = gradZeroMean * alpha
                # SNR dB
                expert.snrdB[c, i] = expert.snrdB[c, i] + 10 * np.log10(
                    (np.sum(np.sum(meas ** 2)) + alpha ** 2 + (sumG * alpha) ** 2) / (s + 2))

            # --- Sum the gradients of all workers
            gradient = gradient + meas

            # === How many indices are active
        # Find the number
        tempTable = np.nonzero(table)
        # Let expert to store it
        expert.activeIdx[c, i] = tempTable[0].size
        # print('Num of Active Indices')
        # print(tempTable[0].size)

        # ================================================= Decoding ================================================= #
        if NOISE:
            # --- Add noise
            if POLAR_SCHEME:
                noise = np.random.normal(loc=0, scale=sigma, size=(gradient.shape[0], gradient.shape[1]))
            else:
                noise = np.random.normal(loc=0, scale=sigma, size=(gradient.shape[0]))

            gradient = (gradient + noise)

            # --- Add noise to two the first part of the channel input ( Parameters, alpha and beta )
            noise = np.random.normal(loc=0, scale=sigma, size=2)
            alphas = alphas + noise[0]
            mu = mu + noise[1]

            # --- Add the mean and scale
            gradient = (gradient + mu) / alphas

        # Use Polar Scheme to decode the indices
        if POLAR_SCHEME:
            # --- Decompress
            numIdx, gradientHat, idxHat, iterations = scheme.recover(gradient, polar)


        if not POLAR_SCHEME:  # and (not NOISE):
            # If you don't use polar scheme, just take the sum of the  gradients and divide by W
            gradientHat = gradient / W

        # ================================================= Check Performance ================================================= #
        if POLAR_SCHEME:
            # True Active indices
            idx = (-table).argsort()[:K]
            idxHat = idxHat[0:numIdx]
            for f in idxHat:
                if f in idx:
                    expert.detection[c, i] = expert.detection[c, i] + 1
                else:
                    expert.falseAlarm[c, i] = expert.falseAlarm[c, i] + 1
                    if f in tempTable[0]:
                        expert.deInFA[c, i] = expert.deInFA[c, i] + 1

            # --- Store Data
            expert.store(c, i, numIdx, gradientHat, idxHat, idx, originalGrad, iterations, s)

        # ================================================= Update Parameters ================================================= #

        # --- Call Server to take the step
        server.step(gradientHat)

        # --- Next communication round
        c += 1

        # --- Find the Test Error at the end of the epoch
        if c == C:
            expert.testError[countTr] = server.checkAcc(xTest, yTest, server.model, device, dtype)
            countTr += 1

    endE = time.time()
    # ================================================= Store Data ================================================= #
    if i % 50 == 0:
        if SAVE == 1:
            expert.saveToCsv()

    # ================================================= End of the Epoch ================================================= #
    print("Epoch Duration")
    print(endE - startE)

# ================================================= End of the Simulation ================================================= #
print("End of the simulation")
print("Total time to run")
print(time.time() - startS)

# ========================================== Save the Final results ===================================================== #
if SAVE == 1:
    expert.saveToCsv()


