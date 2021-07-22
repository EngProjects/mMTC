import numpy as np
import FactorGraphGeneration as FGG
from pyfht import block_sub_fht

def sparc_codebook(L, M, n):
    """
    Create sensing matrix via randomly sampling the rows of Hadamard matrices
    :param L: number of sections
    :param M: length of each section
    :param n: number of channel uses (real d.o.f.)
    """
    Ax, Ay, _ = block_sub_fht(n, M, L, seed=None, ordering=None) # seed must be explicit
    def Ab(b):
        return Ax(b).reshape(-1, 1) / np.sqrt(n)
    def Az(z):
        return Ay(z).reshape(-1, 1) / np.sqrt(n) 
    return Ab, Az

def approximateVector(x, K):   
    """
    Approximate factor graph message by enforcing certain constraints. 
    :param x: vector to approximate
    :param K: number of users in bin
    """ 

    # normalize initial value of x
    xOrig = x / np.linalg.norm(x, ord=1)
    
    # create vector to hold best approximation of x
    xHt = xOrig.copy()
    u = np.zeros(len(xHt))
    
    # run approximation algorithm
    while np.amax(xHt) > (1/K):
        minIndices = np.argmin([(1/K)*np.ones(xHt.shape), xHt], axis=0)
        xHt = np.min([(1/K)*np.ones(xHt.shape), xHt], axis=0)
        
        deficit = 1 - np.linalg.norm(xHt, ord=1)
        
        if deficit > 0:
            mIxHtNorm = np.linalg.norm((xHt*minIndices), ord=1)
            scaleFactor = (deficit + mIxHtNorm) / mIxHtNorm
            xHt = scaleFactor*(minIndices*xHt) + (1/K)*(np.ones(xHt.shape) - minIndices)

    # return admissible approximation of x
    return xHt

def pme0(q, r, d, tau):
    """
    Posterior mean estimator (PME)
    :param q: prior probability
    :param r: effective observation
    :param d: signal amplitude
    :param tau: noise standard deviation
    """
    sHat = ( q*np.exp( -(r-d)**2 / (2*(tau**2)) ) \
            / ( q*np.exp( -(r-d)**2 / (2*(tau**2))) + (1-q)*np.exp( -r**2 / (2*(tau**2))) ) ).astype(float)
    return sHat

def dynamicDenoiser(r,OuterCode,K,tau,d,numBPiter):
    """
    Dynamic AMP denoiser that runs BP on outer factor graph
    :param r: AMP residual
    :param OuterCode: factor graph used in outercode
    :param K: number of users in a bin
    :param tau: noise standard deviation
    :param d: power allocated to each symbol
    :param numBPIter: number of BP iterations to perform on outer factor graph
    """

    # Compute relevant parameters
    M = OuterCode.sparseseclength
    L = OuterCode.varcount
    p0 = 1-(1-1/M)**K
    p1 = p0*np.ones(r.shape, dtype=float)
    mu = np.zeros(r.shape, dtype=float)

    # Compute local estimate (lambda) based on effective observation using PME.
    localEstimates = pme0(p0, r, d, tau)
    
    # Reshape local estimate (lambda) into an LxM matrix
    Beta = localEstimates.reshape(L,-1)

    # Initialize outer factor graph
    OuterCode.reset()
    for varnodeid in OuterCode.varlist:
        idx = varnodeid - 1
        Beta[idx,:] = approximateVector(Beta[idx,:], K)
        OuterCode.setobservation(varnodeid, Beta[idx,:])
    
    # Run BP on outer factor graph
    for iteration in range(numBPiter):
        OuterCode.updatechecks()
        OuterCode.updatevars()

    # Extract and return pertinent information from outer factor graph
    for varnodeid in OuterCode.varlist:
        idx = varnodeid - 1
        Beta[idx,:] = OuterCode.getextrinsicestimate(varnodeid)
        mu[idx*M:(idx+1)*M] = 1 - (1 - approximateVector(Beta[idx,:], K).reshape(-1,1))**K

    return mu

def amp_state_update(z, s, P, L, Az, K, numBPiter, OuterCode):
    """
    Update state for a specific bin.
    :param z: AMP residual
    :param s: current AMP state
    :param P: power allocated to each symbol
    :param L: number of sections in inner code
    :param Az: transpose of the sensing matrix
    :param K: number of users in the specified bin
    :param numBPiter: number of BP iterations to perform on the outer factor graph
    :param OuterCode: factor graph used as the outer code
    """

    # Compute relevant parameters
    n = z.size
    d = np.sqrt(n*P/L)
    z = z.flatten()

    # Compute tau online using the residual
    tau = np.sqrt(np.sum(z**2)/n)

    # Compute effective observation
    r = (d*s + Az(z))

    # Compute updated state
    if K != 0:
        mu = dynamicDenoiser(r, OuterCode, K, tau, d, numBPiter)
    else:
        mu = 0
    s = pme0(mu, r, d, tau)
        
    return s

def amp_residual(y, z, sList, d, AbList):
    """
    Compute residual for use within AMP iterate
    :param y: original observation
    :param z: AMP residual from the previous iteration
    :param sList: list of state updates through AMP composite iteration
    :param d: power allocated to each section
    :param AbList: list of matrices used for inner encoding
    """
    n = y.size
    
    # Compute tau online using the residual
    tau = np.sqrt(np.sum(z**2)/n)

    # Compute residual
    z_plus = y.copy()
    for i in range(NUM_BINS):
        z_plus += -1*d*AbList[i](sList[i]) + (z/(n*tau**2))*((d**2)*(np.sum(sList[i]) - np.sum(sList[i]**2)))
    
    return z_plus

def estimate_bin_occupancy(Ka, dbid, rxK, K, s_n, GENIE_AIDED):
    """
    Estimate the number of users present in each bin. 
    :param Ka: total number of active users
    :param dbid: power allocated to bin occupancy estimation
    :param rxK: received vector indicating how many users are present in each bin
    :param K: true vector of # users/bin
    :param s_n: noise standard deviation
    :param GENIE_AIDED: boolean flag whether to return genie-aided estimates
    """

    # Set up MMSE Estimator
    pi = 1 / NUM_BINS                      # probability of selecting bin i (all bins are equally likely)
    Rzz = s_n**2 * np.eye(NUM_BINS)        # noise autocorrelation matrix
    Rbb = np.zeros((NUM_BINS, NUM_BINS))   # construct autocorrelation matrix for bin identification sequence
    for i in range(NUM_BINS):
        for j in range(NUM_BINS):
            Rbb[i, j] = Ka*pi*(1-pi) if i == j else -1*Ka*pi**2  # variance if diagonal entry, covariance if off diagonal
            Rbb[i, j] += (Ka * pi)**2      # add u_i*u_j to each entry to convert from covariance to autocorrelation
    print('Rbb: ')
    print(Rbb)
    
    # Construct MMSE matrix W
    Ryy = dbid**2 * Rbb + Rzz                           # autocorrelation of rx vector y (A matrix is identity)
    Ryb = dbid * Rbb                                    # cross-correlation of rx vector y and tx vector b 
    W_mmse = np.matmul(np.linalg.inv(Ryy), Ryb)         # LMMSE matrix W

    # LMMSE estimation
    Kht_mmse = np.matmul(W_mmse.conj().T, rxK)          # LMMSE estimate of binIdBits
    mmse = Kht_mmse*Ka/np.linalg.norm(Kht_mmse, ord=1)  # scale estimates to have correct L1 norm 
    Kht = np.maximum(1, np.ceil(mmse)).astype(int)      # take max of 1 and ceil of estimates
    print('True K: \t' + str(K))
    print('Estimated K: \t' + str(Kht))
    
    # Invoke Genie assistance if desired
    if GENIE_AIDED:
        Kht = K.copy()
        print('Genie-aided K: \t' + str(Kht))

    return Kht

def simulate(Ka, NUM_BINS, EbNodB, GENIE_AIDED, ENFORCE_CRC):
    """
    Run coded demixing simulation
    :param Ka: total number of users
    :param NUM_BINS: total number of bins
    :param EbNodB: Eb/No in dB
    :param GENIE_AIDED: flag of whether to use genie-aided estimate of K
    :param ENFORCE_CRC: flag of whether to enforce CRC consistency in recovered codewords
    """

    B = 128                             # length of each user's message in bits
    L = 16                              # number of sections 
    M = 2**L                            # length of each section
    n = 38400                           # number of channel uses (real dof)
    numAMPIter = 10                     # Number of AMP iterations
    numBPiter = 1                       # Number of BP iterations on outer code
    simCount = 100                      # number of trials to average over
    errorRate = 0                       # store error rate
    delta = 5                           # constant number of extra codewords to retain

    # Compute signal and noise power parameters
    EbNo = 10**(EbNodB/10)
    P = 2*B*EbNo/n
    s_n = 1
    
    # Assign power to occupancy estimation and data transmission tasks
    pM = 80
    dcs = np.sqrt(n*P*n/(pM*NUM_BINS + n)/L) if NUM_BINS > 1 else np.sqrt(n*P/L)
    dbid = np.sqrt(n*P*pM*NUM_BINS/(pM*NUM_BINS + n)/NUM_BINS) if NUM_BINS > 1 else 0
    assert np.abs(L*dcs**2 + NUM_BINS*dbid**2 - n*P) <= 1e-3, "Total power constraint violated."

    # run simCount trials
    for simIndex in range(simCount):
        print('**********Simulation Number: ' + str(simIndex))
        
        """*********************************************************************************
        Step 1: users generate messages and stochastically partition themselves into groups. 
        **********************************************************************************"""

        # Generate messages for all Ka users
        usrmessages = np.random.randint(2, size=(Ka, B))

        # Split users into bins based on the first couple of bits in their messages
        w0 = int(np.ceil(np.log2(NUM_BINS)))
        binIds = np.matmul(usrmessages[:,0:w0], 2**np.arange(w0)[::-1]) if w0 > 0 else np.zeros(Ka)
        K = np.array([np.sum(binIds == i) for i in range(NUM_BINS)]).astype(int)

        # Group messages by bin
        messages = [usrmessages[np.where(binIds == i)[0]] for i in range(NUM_BINS)]

        """***************************************************************
        Step 2: receiver estimates the number of users present in each bin
        ***************************************************************"""

        # Transmit bin identifier across channel
        rxK = dbid * K + np.random.randn(NUM_BINS) * s_n

        # Perform LMMSE bin occupancy estimation
        Kht = estimate_bin_occupancy(Ka, dbid, rxK, K, s_n, GENIE_AIDED) if NUM_BINS > 1 else K.copy()
        
        """*******************************************************************
        Step 3: outer/inner message encoding and transmission over AWGN channel
        ********************************************************************"""

        # Generate outer graphs
        OuterCodes = []
        for i in range(NUM_BINS):
            OuterCodes.append(FGG.Triadic8(16))

        # Define data structures to hold encoding/decoding parameters
        txcodewords = 0                                     # list of all codewords transmitted
        codewords = []                                      # list of codewords to transmitted by bin
        sTrue = []                                          # list of signals sent by various bins
        Ab = []                                             # list of sensing matrices used by various bins
        Az = []                                             # list of sensing matrices transposed used by various bins

        # Bin-Specific Encoding Operations
        for i in range(NUM_BINS):

            # Outer encode each message
            cdwds = OuterCodes[i].encodemessages(messages[i])
            for cdwd in cdwds:                              # ensure that each codeword is valid
                OuterCodes[i].testvalid(cdwd)
            codewords.append(cdwds)                         # add encoded messages to list of codewords
            txcodewords = np.vstack((txcodewords, cdwds)) if not np.isscalar(txcodewords) else cdwds.copy()

            # Combine codewords to form signal to transmit
            if K[i] == 0:                                   # do nothing if there are no users in this bin
                tmp = np.zeros(L*M).astype(np.float64)
            else:                                           # otherwise, add all codewords together
                tmp = np.sum(cdwds, axis=0)
            sTrue.append(tmp)                               # store true signal for future reference

            # Generate the binned SPARC codebook
            a, b = sparc_codebook(L, M, n)
            Ab.append(a)
            Az.append(b)

        # Generate transmitted signal X
        x = 0.0
        for i in range(NUM_BINS):
            x += dcs*Ab[i](sTrue[i])
        
        # Transmit signal X through AWGN channel
        y = x + np.random.randn(n, 1) * s_n
        
        """***********************
        Step 4: Inner AMP decoding
        ***********************"""

        # Prepare for inner AMP decoding
        z = y.copy()                                        # initialize AMP residual
        s = [np.zeros((L*M, 1)) for i in range(NUM_BINS)]   # initialize AMP states

        # AMP Inner decoder
        for idxampiter in range(numAMPIter):

            # Update the state of each bin individually
            s = [amp_state_update(z, s[i], P, L, Az[i], Kht[i], numBPiter, OuterCodes[i]) for i in range(NUM_BINS)]

            # compute residual jointly
            z = amp_residual(y, z, s, dcs, Ab)

        """*************************
        Step 5: Outer graph decoding
        *************************"""
        
        # Prepare for outer graph decoding
        recoveredcodewords = dict()     

        # Graph-based outer decoder
        for idxbin in range(NUM_BINS):
            if Kht[idxbin] == 0: continue

            # Produce list of recovered codewords with their associated likelihoods
            recovered, likelihoods = OuterCodes[idxbin].decoder(s[idxbin], int(Kht[idxbin] + delta), includelikelihoods=True)

            # Compute what the first w0 bits should be based on bin number
            binIDBase2 = np.binary_repr(idxbin)
            binIDBase2 = binIDBase2 if len(binIDBase2) == w0 else (w0 - len(binIDBase2))*'0' + binIDBase2
            firstW0bits = np.array([binIDBase2[i] for i in range(len(binIDBase2))]).astype(int)

            # Add recovered codewords to data structure indexed by likelihood with optionally enforced CRC check
            for idxcdwd in range(len(likelihoods)):
                if (not ENFORCE_CRC) or (NUM_BINS == 1):
                    recoveredcodewords[likelihoods[idxcdwd]] = recovered[idxcdwd]
                else:
                    # Extract first part of message from codeword
                    firstinfosection = OuterCodes[idxbin].infolist[0] - 1
                    sparsemsg = recovered[idxcdwd][firstinfosection*M:(firstinfosection+1)*M]

                    # Find index of nonzero entry and convert to binary representation
                    idxnonzero = np.where(sparsemsg > 0.0)[0][0]
                    idxnonzerobin = np.binary_repr(idxnonzero)

                    # Add trailing zeros to base-2 representation
                    if len(idxnonzerobin) < 16:
                        idxnonzerobin = (16 - len(idxnonzerobin))*'0' + idxnonzerobin

                    # Extract first w0 bits
                    msgfirstW0bits = np.array([idxnonzerobin[i] for i in range(w0)]).astype(int)

                    # Enforce CRC consistency
                    if (msgfirstW0bits==firstW0bits).all():
                        recoveredcodewords[likelihoods[idxcdwd]] = recovered[idxcdwd]
            
        # sort dictionary of recovered codewords in descending order of likelihood
        sortedcodewordestimates = sorted(recoveredcodewords.items(), key=lambda x: -x[0])[0:Ka]
        codewordestimates = 0
        for idxusr in range(len(sortedcodewordestimates)):
            codewordestimates = np.vstack((codewordestimates, sortedcodewordestimates[idxusr][1])) if not np.isscalar(codewordestimates) else \
                                sortedcodewordestimates[idxusr][1].copy()

        """*****************
        Step 6: Compute PUPE
        *****************"""

        # Compute error rate
        matches = FGG.numbermatches(txcodewords, codewordestimates)
        errorRate += (Ka - matches) / (Ka * simCount)
        print(str(matches) + ' matches')
        print('Cumulative Error Rate: ' + str(errorRate*simCount/(simIndex+1)))

    return errorRate

# Simulation parameters
Ka = 64                             # total number of users
NUM_BINS = 2                        # number of bins employed in simulation
GENIE_AIDED = False                 # flag of whether to produce genie-aided bin occupancy estimates
ENFORCE_CRC = False                 # flag of whether to enforce CRC consistency condition during decoding
SNRs = [1.6, 1.8, 2.0, 2.2, 2.4]    # EbNo values to simulate
errorRates = []                     # data structure to store error results

# Run simulation
for snr in SNRs:
    print('SNR = ' + str(snr))
    a = simulate(Ka, NUM_BINS, snr, GENIE_AIDED, ENFORCE_CRC)

    print('*****************************************************************************************')
    print('SNR: ' + str(snr) + ' Error rate: ' + str(a))
    print('*****************************************************************************************')
    
    np.savetxt(str(NUM_BINS)+'_'+str(snr)+'_pupe.txt', np.array([a]))
    errorRates.append(a)

# Print and store simulation results
print('Simulation complete!')
print(errorRates)
filename = str(NUM_BINS) + '_bins_pupe_vs_snr.txt'
np.savetxt(filename, errorRates)