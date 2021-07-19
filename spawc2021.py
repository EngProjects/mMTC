import numpy as np
import FactorGraphGeneration as FGG
from pyfht import block_sub_fht

NUM_BINS = 2

# Generate outer graphs
OuterCodes = []
for i in range(NUM_BINS):
    OuterCodes.append(FGG.Triadic8(16))

def binID(binNum, NUM_BINS):
    assert binNum < NUM_BINS, "Invalid binNum. "
    binid = np.zeros(NUM_BINS)
    binid[binNum] = 1
    return binid

def sparc_codebook(L, M, n,P):
    Ax, Ay, _ = block_sub_fht(n, M, L, seed=None, ordering=None) # seed must be explicit
    def Ab(b):
        return Ax(b).reshape(-1, 1) / np.sqrt(n)
    def Az(z):
        return Ay(z).reshape(-1, 1) / np.sqrt(n) 
    return Ab, Az

def approximateVector(x, K):    

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
    """Posterior mean estimator (PME)
    
    Args:
        q (float): Prior probability
        r (float): Effective observation
        d (float): Signal amplitude
        tau (float): Standard deviation of noise
    Returns:
        sHat (float): Probability s is one
    
    """
    sHat = ( q*np.exp( -(r-d)**2 / (2*(tau**2)) ) \
            / ( q*np.exp( -(r-d)**2 / (2*(tau**2))) + (1-q)*np.exp( -r**2 / (2*(tau**2))) ) ).astype(float)
    return sHat

def dynamicDenoiser(r,OuterCode,K,tau,d,numBPiter):
    """
    Args:
        r (float): Effective observation
        d (float): Signal amplitude
        tau (float): Standard deviation of noise
    """
    M = OuterCode.sparseseclength
    L = OuterCode.varcount

    p0 = 1-(1-1/M)**K
    p1 = p0*np.ones(r.shape, dtype=float)
    mu = np.zeros(r.shape, dtype=float)

    # Compute local estimate (lambda) based on effective observation using PME.
    localEstimates = pme0(p0, r, d, tau)
    
    # Reshape local estimate (lambda) into an LxM matrix
    Beta = localEstimates.reshape(L,-1)
    OuterCode.reset()
    for varnodeid in OuterCode.varlist:
        idx = varnodeid - 1
        Beta[idx,:] = approximateVector(Beta[idx,:], K)
        OuterCode.setobservation(varnodeid, Beta[idx,:])
    
    for iteration in range(numBPiter):
        OuterCode.updatechecks()
        OuterCode.updatevars()

    for varnodeid in OuterCode.varlist:
        idx = varnodeid - 1
        # Beta[idx,:] = OuterCode.getestimate(varnodeid)
        Beta[idx,:] = OuterCode.getextrinsicestimate(varnodeid)
        mu[idx*M:(idx+1)*M] = 1 - (1 - approximateVector(Beta[idx,:], K).reshape(-1,1))**K

    return mu

def amp_state_update(z, s, P, L, Ab, Az, K, numBPiter, OuterCode):

    """
    Args:
        s: State update through AMP composite iteration
        z: Residual update through AMP composite iteration
        tau (float): Standard deviation of noise
        mu: Product of messages from adjoining factors
    """
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
    Args:
        s1: State update through AMP composite iteration
        s2: State update through AMP composite iteration
        y: Original observation
        tau (float): Standard deviation of noise
    """
    n = y.size
    
    # Compute tau online using the residual
    tau = np.sqrt(np.sum(z**2)/n)

    # Compute residual
    z_plus = y.copy()

    for i in range(NUM_BINS):
        z_plus += -1*d*AbList[i](sList[i]) + (z/(n*tau**2))*((d**2)*(np.sum(sList[i]) - np.sum(sList[i]**2)))
    
    return z_plus

def simulate(EbNodB):
    Ka = 64
    B = 128
    L = OuterCodes[0].varcount
    M = OuterCodes[0].sparseseclength
    n = 38400      # number of channel uses (real dof)
    T = 10         # Number of AMP iterations
    numBPiter = 1  # Number of BP iterations on outer code. 1 seems to be good enough & AMP theory including state evolution valid only for one BP iteration
    simCount = 100 # number of simulations
    errorRate = 0  # store error rate
    delta = 5      # constant number of extra codewords to retain = std(#users per bin)

    # EbN0 in linear scale
    EbNo = 10**(EbNodB/10)
    P = 2*B*EbNo/n
    s_n = 1
    
    # Compute power multiplier for bin identifier bits
    pM = 80
    dcs = np.sqrt(n*P*n/(pM*NUM_BINS + n)/L)
    dbid = np.sqrt(n*P*pM*NUM_BINS/(pM*NUM_BINS + n)/NUM_BINS)
    assert np.abs(L*dcs**2 + NUM_BINS*dbid**2 - n*P) <= 1e-3, "Total power constraint violated."
    
    # MMSE Estimator parameters
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
    Ryy = dbid**2 * Rbb + Rzz              # autocorrelation of rx vector y (A matrix is identity)
    Ryb = dbid * Rbb                       # cross-correlation of rx vector y and tx vector b 
    W_mmse = np.matmul(np.linalg.inv(Ryy), Ryb)         # LMMSE matrix W

    for simIndex in range(simCount):
        print('**********Simulation Number: ' + str(simIndex))
        
        # Randomly assign users to groups
        K = np.zeros(NUM_BINS).astype(int)
        for i in range(Ka):
            K[np.random.randint(NUM_BINS)] += 1
        
        # Define bin LUT
        binIdLUT = np.eye(NUM_BINS) * dbid

        # Send bin identifiers
        binIdBits = np.matmul(K, binIdLUT)

        # Transmit bin identifier across channel
        rxBinIdBits = binIdBits + np.random.randn(NUM_BINS) * s_n
        
        # LMMSE estimation
        Kht = np.zeros(NUM_BINS).astype(int)   # data structure to store result in 
        Kht_mmse = np.matmul(W_mmse.conj().T, rxBinIdBits)  # LMMSE estimate of binIdBits
        mmse = Kht_mmse*Ka/np.linalg.norm(Kht_mmse, ord=1)  # scale estimates to have correct L1 norm 
        Kht = np.maximum(1, np.ceil(mmse)).astype(int)      # take max of 1 and ceil of estimates
        
        # Print estimation results
        print('True K: \t' + str(K))
        print('Estimated K: \t' + str(Kht))
        
        #######################################################################################################
        # GENERATE GENIE-AIDED CURVE BELOW
        # Kht = K.copy()
        ########################################################################################################
        
        # Generate active users message sequences
        messages = []
        for i in range(NUM_BINS):
            messages.append(np.random.randint(2, size=(K[i], B)))

        # Outer-encode the message sequences
        codewordmasterlist = -1*np.ones(L*M + NUM_BINS)
        codewords = []
        sTrue = []
        Ab = []
        Az = []
        for i in range(NUM_BINS):
            # Outer-encode the message sequences
            cdwds = OuterCodes[i].encodemessages(messages[i])
            for cdwd in cdwds:
                OuterCodes[i].testvalid(cdwd)
            codewords.append(OuterCodes[i].encodemessages(messages[i]))
            for j in range(K[i]):
                codewordmasterlist = np.vstack((
                    codewordmasterlist, 
                    np.hstack((binID(i, NUM_BINS), codewords[i][j,:]))
                ))

            # Convert indices to sparse representation
            if K[i] == 0:
                tmp = np.zeros(L*M).astype(np.float64)
            else:
                tmp = np.sum(cdwds, axis=0)
            sTrue.append(tmp)

            # Generate the binned SPARC codebook
            a, b = sparc_codebook(L, M, n, P)
            Ab.append(a)
            Az.append(b)
        codewordmasterlist = codewordmasterlist[1:, :]

        # Generate our transmitted signal X
        x = 0.0
        for i in range(NUM_BINS):
            x += dcs*Ab[i](sTrue[i])
        
        # Generate random channel noise and thus also received signal y
        noise = np.random.randn(n, 1) * s_n
        y = (x + noise)
        
        # Receiver processing
        z = y.copy()
        
        s = []
        for i in range(NUM_BINS):
            s.append(np.zeros((L*M, 1)))

        for t in range(T):
            print(np.sqrt(np.sum(z**2)/n))
            for i in range(NUM_BINS):
                s[i] = amp_state_update(z, s[i], P, L, Ab[i], Az[i], Kht[i], numBPiter, OuterCodes[0])
            z = amp_residual(y, z, s, dcs, Ab)
        
        # Decoding with Graph
        print('Graph Decode')
        recoveredcodewords = dict()
        for i in range(NUM_BINS):
            if Kht[i] == 0:
                continue
            recovered, likelihoods = OuterCodes[i].decoder(s[i], int(Kht[i]*2 + delta), includelikelihoods=True)
            for idxcdwd in range(len(likelihoods)):
                recoveredcodewords[likelihoods[idxcdwd]] = np.hstack((binID(i, NUM_BINS), recovered[idxcdwd]))
            
        # sort dictionary of recovered codewords in descending order of likelihood
        sortedcodewordestimates = sorted(recoveredcodewords.items(), key=lambda x: -x[0])
        sortedcodewordestimates = sortedcodewordestimates[0:Ka]
        assert len(sortedcodewordestimates) == Ka, "Retained more than Ka codeword estimates"
        codewordestimates = -1*np.ones(L*M + NUM_BINS)
        for idxi in range(Ka):
            codewordestimates = np.vstack((codewordestimates, sortedcodewordestimates[idxi][1]))
        codewordestimates = codewordestimates[1:]
        
        # Compute error rate
        matches = FGG.numbermatches(codewordmasterlist, codewordestimates)
        errorRate += ((Ka - matches)/Ka)/simCount
        print(str(matches) + ' matches')
        print('Cumulative Error Rate: ' + str(errorRate*simCount/(simIndex+1)))

    return errorRate


SNRs = [1.6, 1.8, 2.0, 2.2, 2.4]
errorRates = []

for snr in SNRs:
    print('SNR = ' + str(snr))
    a = simulate(snr)
    print('*****************************************************************************************')
    print('SNR: ' + str(snr) + ' Error rate: ' + str(a))
    print('*****************************************************************************************')
    np.savetxt(str(NUM_BINS)+'_'+str(snr)+'_pupe.txt', np.array([a]))
    errorRates.append(a)

print(errorRates)
filename = str(NUM_BINS) + '_bins_pupe_vs_snr.txt'
np.savetxt(filename, errorRates)