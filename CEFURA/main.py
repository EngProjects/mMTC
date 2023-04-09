import time
import numpy as np
import matplotlib.pyplot as plt
from Network import Network
from CPU import CPU
from UserEquipment import UEs
from AccessPoint import APs
np.random.seed(0)

# ====================================== Initialization ====================================== #
# --- Simulation
debug = 0
plot = 0
# --- Wireless Network
D = 550
nUEsArray = np.array([50, 75, 100, 125, 150])
nAPs = 1
nAnts = 100

# ============= Cluster Point Process Info
type = 'B' # B: Binomial Point Process, P: Poisson Point Process  T: Thomas Cluster Point Process, M:  Matérn Cluster Point Process
radius = 80 # For M
sigma2P = 729 # For T
nClusters = 25 
densityP = nClusters / (D ** 2)
densityD = nUEsArray[0] / nClusters



# --- Noise
sigma2 = 3.9810717055e-12 # -84dBm


# --- Coding Scheme
# number of channel uses
nChanlUses = 3200
# Spreading sequence length
L = 10
# Length of the code
nc = 512
# number of pilots
nPilots = int(nChanlUses - L *(nc/2))
# Length of the message, Length of the First and Second part of the message
B, Bf = 100, 15
Bs = B - Bf
# Number of spreading sequence/ pilot sequence
J = 2 ** Bf
# Length of List (Decoder)
nL = 8
# Number of channel uses
nChanlUses = int((nc / np.log2(4))*L + nPilots)
# User transmit power
Pt = 10e-3


# --- To store results
nIter = 150
nUEsPoints = len(nUEsArray)
iterRange = np.arange(nIter) + 1
probDE = np.zeros((nUEsPoints, nIter))
probFA = np.zeros((nUEsPoints, nIter))
# ====================================== Simulation ====================================== #
start = time.time()
print("# ============= Simulation ============= #")
print('Number of Access Points: ' + str(nAPs))
print('Number of Antennas: ' + str(nAnts))
print("Number of channel uses: " +str(nChanlUses))
print("Number of iterations: "+ str(nIter))
print("Length of spreading sequence " + str(L))
print("Length of the code " +str(nc))
print("Length of pilots " + str(nPilots))
print("Length of message bits = " +str(B))
print("Bf = " + str(Bf))
print('Tx SNR (int): ' + str(int(10*np.log10(Pt/sigma2))) + 'dB')
print('Tx power : ' + str(Pt) + 'W')
print('Cell height/width : ' + str(D) + 'm')
print('Distribution Type: ' + type)



indicator = 0
for nUEs in nUEsArray:
    if nAPs == 1:
        nRec = nUEs
    else:
        nRec = 7

    if type != 'B':
        if type == 'P':
            densityP = nUEs / (D ** 2)
        else:
            densityP = nClusters / (D ** 2)
        densityD = nUEs / nClusters

    print('========================== Number of users: ' + str(nUEs) + ' ==========================')
    # --- Create a network object
    net = Network(D, nAPs, nAnts, nUEs, type, densityP, densityD, radius, sigma2P)


    # --- Create a CPU object
    cpu = CPU(nUEs, nAPs, B, Bf, nc, nL, sigma2)

    error = 0
    for Iter in range(nIter):

        # --- Generate Users at Random
        net.generateUsers()
        if type != 'B':
            # --- Create a CPU object
            nUEs = net.nUEs
            cpu = CPU(nUEs, nAPs, B, Bf, nc, nL, sigma2)

        if 0 and Iter == 0:
            net.plotCell()

        # --- Generate nUEs msgs
        msgs = np.random.randint(0, 2, (nUEs, B))

        # --- Generate the channel coefficients
        H = net.generateChannel()

        # --- Generate the noise
        N = (1 / np.sqrt(2)) * (np.random.normal(0, np.sqrt(sigma2), (nChanlUses, nAPs * nAnts)) + 1j * np.random.normal(0, np.sqrt(sigma2), (nChanlUses, nAPs * nAnts)))

        # --- Create an User Equipment object
        users = UEs(nUEs, nAPs, nAnts, B, Bf, L, nc, nL, nPilots, sigma2, Pt)

        # --- Transmit data
        HX = users.transmit(msgs, H)
        Y = HX + N

        # --- Create an Access Points object
        accessPoints = APs(nUEs, nAPs, nAnts, B, Bf, L, nc, nL, nPilots, sigma2, users, nRec)

        # --- APs receiver
        symbolsHat, idxSSHat = accessPoints.receiver(Y)

        # --- CPU
        nDE, nFA, recnUEs = cpu.combine(symbolsHat, idxSSHat, users.frozenValues,  users.msgs, users.idx2UE, users.codewords)

        if recnUEs > 0:
          probFA[indicator, Iter] = nFA/recnUEs
        probDE[indicator, Iter] = nDE/nUEs


    print('------------------ Results ------------------')
    print('Missed Detection')
    print(np.sum(1 - probDE, 1) / float(nIter))
    print('False Alarm')
    print(np.sum(probFA, 1) / float(nIter))
    indicator += 1


print(time.time() - start)
print()
