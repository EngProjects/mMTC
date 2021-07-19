import FactorGraphGeneration as FG
import ccsinnercode as ccsic
import numpy as np
import time 

# Simulation Parameters
K = 100                                             # number of active users
N = 38400                                           # number of channel uses (real d.o.f)
w = 128                                             # length of each user's uncoded message
numAMPIter = 10                                     # number of AMP iterations to perform
numBPIter = 1                                       # number of BP iterations to perform
listSize = K + 10                                   # list size retained per section after AMP converges
nstd = 1                                            # AWGN noise standard deviation
numSims = 2                                       # number of trials to average over per point on the graph
OuterGraph = FG.Triadic8(16)                        # factor graph associated with outer LDPC code
SNRs = np.arange(start=1.5, stop=4.6, step=0.5)     # SNRs in (dB) to test over
SNRs = np.array([2.5])
errorrates = np.zeros((4, len(SNRs)))               # data structure for storing PUPE results
runtimes = np.zeros((4, len(SNRs)))                 # data structure for storing runtime results

# Iterate over each SNR
for idxsnr in range(len(SNRs)):

    # Compute power
    EbNo = 10**(SNRs[idxsnr]/10)
    P = 2*w*EbNo/N

    # Average over maxSims trials
    for idxsim in range(numSims):

        # Reset the graph
        OuterGraph.reset()

        # Set up Inner Encoder/Decoders
        # Case 1: Original CCS: block diagonal sensing matrix, no BP on outer graph
        # Case 2: CCS-AMP with no BP: dense sensing matrix, no BP on outer graph
        # Case 3: CCS-AMP with BP: dense sensing matrix, BP on outer graph
        # Case 4: CCS-Hybrid: block diagonal sensing matrix, BP on outer graph
        c1InnerCode = ccsic.BlockDiagonalInnerCode(N, P, nstd, K, OuterGraph)
        c2InnerCode = ccsic.DenseInnerCode(N, P, nstd, K, OuterGraph)
        c3InnerCode = ccsic.DenseInnerCode(N, P, nstd, K, OuterGraph)
        c4InnerCode = ccsic.BlockDiagonalInnerCode(N, P, nstd, K, OuterGraph)

        # Generate random messages for K active users
        txBits = np.random.randint(low=2, size=(K, w))

        # Outer LDPC Encoder
        txMsg = OuterGraph.encodemessages(txBits)
        x = OuterGraph.encodesignal(txBits)

        # Inner CS Encoder
        x1 = c1InnerCode.Encode(x)
        x2 = c2InnerCode.Encode(x)
        x3 = c3InnerCode.Encode(x)
        x4 = c4InnerCode.Encode(x)

        # Transmit over channel
        y1 = (x1 + nstd * np.random.randn(N, 1)).reshape((-1, 1))
        y2 = (x2 + nstd * np.random.randn(N, 1)).reshape((-1, 1))
        y3 = (x3 + nstd * np.random.randn(N, 1)).reshape((-1, 1))
        y4 = (x4 + nstd * np.random.randn(N, 1)).reshape((-1, 1))

        # Case 1 Decoding
        tic = time.time()
        xHt, _ = c1InnerCode.Decode(y1, numAMPIter, BPonOuterGraph=False, numBPIter=0, graph=OuterGraph)
        toc = time.time()
        txMsgHt = OuterGraph.decoder(xHt, listSize)

        runtimes[0, idxsnr] += (toc - tic) / numSims
        errorrates[0, idxsnr] += (K - FG.numbermatches(txMsg, txMsgHt)) / (K * numSims)

        # Case 2 Decoding
        tic = time.time()
        xHt, _ = c2InnerCode.Decode(y2, numAMPIter, BPonOuterGraph=False, numBPIter=0, graph=OuterGraph)
        toc = time.time()
        txMsgHt = OuterGraph.decoder(xHt, listSize)

        runtimes[1, idxsnr] += (toc - tic) / numSims
        errorrates[1, idxsnr] += (K - FG.numbermatches(txMsg, txMsgHt)) / (K * numSims)

        # Case 3 Decoding
        tic = time.time()
        xHt, _ = c3InnerCode.Decode(y3, numAMPIter, BPonOuterGraph=True, numBPIter=numBPIter, graph=OuterGraph)
        toc = time.time()
        txMsgHt = OuterGraph.decoder(xHt, listSize)

        runtimes[2, idxsnr] += (toc - tic) / numSims
        errorrates[2, idxsnr] += (K - FG.numbermatches(txMsg, txMsgHt)) / (K * numSims)

        # Case 4 Decoding
        tic = time.time()
        xHt, _ = c4InnerCode.Decode(y4, numAMPIter, BPonOuterGraph=True, numBPIter=numBPIter, graph=OuterGraph)
        toc = time.time()
        txMsgHt = OuterGraph.decoder(xHt, listSize)

        runtimes[3, idxsnr] += (toc - tic) / numSims
        errorrates[3, idxsnr] += (K - FG.numbermatches(txMsg, txMsgHt)) / (K * numSims)

        print(runtimes)
        print(errorrates)

# Print out results
print('Simulation Complete. ')
print('Error rates: \n' + str(errorrates))
print('Runtimes: \n' + str(runtimes))
