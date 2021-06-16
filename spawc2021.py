import numpy as np
import FactorGraphGeneration as FGG
from pyfht import block_sub_fht

NUM_BINS = 2

# Generate outer graphs
OuterCodes = []
for i in range(NUM_BINS):
    OuterCodes.append(FGG.Triadic8(16))

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
        mu[idx*M:(idx+1)*M] = approximateVector(Beta[idx,:], K).reshape(-1,1)

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
    mu = dynamicDenoiser(r, OuterCode, K, tau, d, numBPiter)
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

    n=38400 # Total number of channel uses (real d.o.f)
    T=10 # Number of AMP iterations
    numBPiter = 1; # Number of BP iterations on outer code. 1 seems to be good enough & AMP theory including state evolution valid only for one BP iteration
    simCount = 1 # number of simulations

    # EbN0 in linear scale
    EbNo = 10**(EbNodB/10)
    P = 2*B*EbNo/n
    σ_n = 1

    # We assume equal power allocation for all the sections. Code has to be modified a little to accomodate non-uniform power allocations
    d = np.sqrt(n*P/L)

    msgDetected = np.zeros(NUM_BINS)
    errorRates = np.zeros(NUM_BINS)

    for simIndex in range(simCount):
        print('******************************Simulation Number: ' + str(simIndex))
        
        # Randomly assign users to each bin
        K = np.zeros(NUM_BINS).astype(int)
        for idxk in range(Ka):
          K[np.random.randint(NUM_BINS)] += 1
        
        # Generate active users message sequences
        messages = []
        for i in range(NUM_BINS):
            messages.append(np.random.randint(2, size=(K[i], B)))

        # Outer-encode the message sequences
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

            # Convert indices to sparse representation
            sTrue.append(np.sum(cdwds, axis=0))

            # Generate the binned SPARC codebook
            a, b = sparc_codebook(L, M, n, P)
            Ab.append(a)
            Az.append(b)

        
        # Generate our transmitted signal X
        x = 0.0
        for i in range(NUM_BINS):
            x += d*Ab[i](sTrue[i])
        # x = d*Ab1(sTrue1) + d*Ab2(sTrue2)
        
        # Generate random channel noise and thus also received signal y
        noise = np.random.randn(n, 1) * σ_n
        y = (x + noise)

        z = y.copy()
        
        s = []
        for i in range(NUM_BINS):
            s.append(np.zeros((L*M, 1)))

        for t in range(T):
            print(np.sqrt(np.sum(z**2)/n))
            for i in range(NUM_BINS):
                s[i] = amp_state_update(z, s[i], P, L, Ab[i], Az[i], K[i], numBPiter, OuterCodes[0])
            z = amp_residual(y, z, s, d, Ab)

        print('Graph Decode')
        
        # Decoding with Graph
        for i in range(NUM_BINS):
            original = codewords[i].copy()
            recovered = OuterCodes[i].decoder(s[i], int(K[i] * 1.5))
            matches = FGG.numbermatches(original, recovered)
            print('Group ' + str(i+1) + ': ' + str(matches) + ' out of ' + str(K[i]))
            msgDetected[i] += matches


    for i in range(NUM_BINS):
        errorRates[i] = (K[i]*simCount - msgDetected[i]) / (K[i] * simCount)
        print("Per user probability of error (Group " + str(i+1) + ") = ", errorRates[i])   

    # return errorRate1, errorRate2, 0.5*(errorRate1 + errorRate2)
    return errorRates, np.average(errorRates)


SNRs = [2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
avgErrorRates = []

for snr in SNRs:
    print('SNR = ' + str(snr))
    a, b = simulate(snr)
    avgErrorRates.append(b)

print(avgErrorRates)
filename = str(NUM_BINS) + '_class_avg_pupe.txt'
np.savetxt(filename, avgErrorRates)