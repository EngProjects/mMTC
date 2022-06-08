#!/usr/bin/env python
# coding: utf-8

# # Coded Demxing
# 
# This notebook implements coded Demixing using the CCS-AMP encoder/decoder for multi-class unsourced random access using Hadamard design matrices.
# 

# In[ ]:


import numpy as np
import math
#import matplotlib.pyplot as plt
import time
import sys
import FactorGraphGeneration as FGG

OuterCode = FGG.Graph62()


# ## Fast Hadamard Transforms
# 
# The ```PyFHT_local``` code can all be found in `pyfht`, which uses a C extension to speed up the fht function.
# Only one import suffices, with the latter being much faster.

# In[ ]:


# import PyFHT_local
from pyfht import block_sub_fht


# # Outer Tree encoder
# 
# This function encodes the payloads corresponding to users into codewords from the specified tree code. 
# 
# Parity bits in section $i$ are generated based on the information sections $i$ is connected to
# 
# Computations are done within the ring of integers modulo length of the section to enable FFT-based BP on the outer graph
# 
# This function outputs the sparse representation of encoded messages

# In[ ]:


def Tree_encode(tx_message,K,messageBlocks,G,L,J):
    encoded_tx_message = np.zeros((K,L),dtype=int)
    
    encoded_tx_message[:,0] = tx_message[:,0:J].dot(2**np.arange(J)[::-1])
    for i in range(1,L):
        if messageBlocks[i]:
            # copy the message if i is an information section
            encoded_tx_message[:,i] = tx_message[:,np.sum(messageBlocks[:i])*J:(np.sum(messageBlocks[:i])+1)*J].dot(2**np.arange(J)[::-1])
        else:
            # compute the parity if i is a parity section
            indices = np.where(G[i])[0]
            ParityInteger=np.zeros((K,1),dtype='int')
            for j in indices:
                ParityInteger = ParityInteger + encoded_tx_message[:,j].reshape(-1,1)
            encoded_tx_message[:,i] = np.mod(ParityInteger,2**J).reshape(-1)
    
    return encoded_tx_message


# This function converts message indices into $L$-sparse vectors of length $L 2^J$.

# In[ ]:


def convert_indices_to_sparse(encoded_tx_message_indices,L,J,K):
    aggregate_state_s_sparse=np.zeros((L*2**J,1),dtype=int)
    for i in range(L):
        section_indices_vectorized_rows = encoded_tx_message_indices[:,i]
        section_indices_vectorized_cols = section_indices_vectorized_rows.reshape([-1,1])
        np.add.at(aggregate_state_s_sparse, (i*2**J)+section_indices_vectorized_cols, 1)

    return aggregate_state_s_sparse


# This function returns the index representation corresponding to a SPARC-like vector.

# In[ ]:


def convert_sparse_to_indices(cs_decoded_tx_message_sparse,L,J,listSize):
    cs_decoded_tx_message = np.zeros((listSize,L),dtype=int)
    for i in range(L):
        aggregate_section_sHat_sparse = cs_decoded_tx_message_sparse[i*2**J:(i+1)*2**J]
        indices_low_values = (aggregate_section_sHat_sparse.reshape(2**J,)).argsort()[np.arange(2**J-listSize)]
        indices_high_values = np.setdiff1d(np.arange(2**J),indices_low_values)
        cs_decoded_tx_message[:,i] = indices_high_values

    return cs_decoded_tx_message


# Extract information bits from retained paths in the tree.

# In[ ]:


def extract_msg_indices(Paths,cs_decoded_tx_message, L,J):
    msg_bits = np.empty(shape=(0,0))
    L1 = Paths.shape[0]
    for i in range(L1):
        msg_bit=np.empty(shape=(0,0))
        path = Paths[i].reshape(1,-1)
        for j in range(path.shape[1]):
            msg_bit = np.hstack((msg_bit,cs_decoded_tx_message[path[0,j],j].reshape(1,-1))) if msg_bit.size else cs_decoded_tx_message[path[0,j],j]
            msg_bit=msg_bit.reshape(1,-1)
        msg_bits = np.vstack((msg_bits,msg_bit)) if msg_bits.size else msg_bit

    return msg_bits


# ## SPARC Codebook
# 
# We use the `block_sub_fht` which computes the equivalent of $A.\beta$ by using $L$ separate $M\times M$ Hadamard matrices. However we want each entry to be divided by $\sqrt{n}$ to get the right variance, and we need to do a reshape on the output to get column vectors, so we'll wrap those operations here.
# 
# Returns two functions `Ab` and `Az` which compute $A\cdot B$ and $z^T\cdot A$ respectively.

# In[ ]:


def sparc_codebook(L, M, n,P):
    Ax, Ay, _ = block_sub_fht(n, M, L, seed=None, ordering=None) # seed must be explicit
    def Ab(b):
        return Ax(b).reshape(-1, 1)/ np.sqrt(n)
    def Az(z):
        return Ay(z).reshape(-1, 1)/ np.sqrt(n) 
    return Ab, Az


# # Vector Approximation
# 
# This function outputs the closest approximation to the input vector given that its L1 norm is 1 and no entry is greater than 1/K

# In[ ]:


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


# ## Posterior Mean Estimator (PME)
# 
# This function implements the posterior mean estimator for situations where prior probabilities are uninformative.

# In[ ]:


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
    sHat = ( q*np.exp( -(r-d)**2 / (2*(tau**2)) )             / ( q*np.exp( -(r-d)**2 / (2*(tau**2))) + (1-q)*np.exp( -r**2 / (2*(tau**2))) ) ).astype(float)
    return sHat


# # Dynamic Denoiser
# 
# This function performs believe propagation (BP) on the factor graph of the outer code.

# In[ ]:


def dynamicDenoiser1(r,G,messageBlocks,L,M,K,tau,d,numBPiter):
    """
    Args:
        r (float): Effective observation
        d (float): Signal amplitude
        tau (float): Standard deviation of noise
    """
    p0 = 1-(1-1/M)**K
    p1 = p0*np.ones(r.shape,dtype=float)
    mu = np.zeros(r.shape,dtype=float)

    # Compute local estimate (lambda) based on effective observation using PME.
    localEstimates = pme0(p0, r, d, tau)
    
    # Reshape local estimate (lambda) into an LxM matrix
    Beta = localEstimates.reshape(L,-1)
    for i in range(L):
        Beta[i,:] = approximateVector(Beta[i,:], K)

    # There is an issue BELOW for numBPiter greater than one!
    for iter in range(numBPiter):    
        # Rotate PME 180deg about y-axis
        Betaflipped = np.hstack((Beta[:,0].reshape(-1,1),np.flip(Beta[:,1:],axis=1)))
        # Compute and store all FFTs
        BetaFFT = np.fft.rfft(Beta)
        BetaflippedFFT = np.fft.rfft(Betaflipped)
        for i in range(L):
            if messageBlocks[i]:
                # Parity sections connected to info section i
                parityIndices = np.where(G[i])[0]   # Identities of parity block(s) attached
                BetaIFFTprime = np.empty((0,0)).astype(float)
                for j in parityIndices:  # Compute message for check associated with parity j
                    # Other info blocks connected to this parity block
                    messageIndices = np.setdiff1d(np.where(G[j])[0],i)  ## all indicies attahced to j, other than i
                    BetaFFTprime = np.vstack((BetaFFT[j],BetaflippedFFT[messageIndices,:]))  ## j is not part of G[j]
                    # Multiply the relevant FFTs
                    BetaFFTprime = np.prod(BetaFFTprime,axis=0)
                    # IFFT
                    BetaIFFTprime1 = np.fft.irfft(BetaFFTprime).real # multiple parity
                    BetaIFFTprime = np.vstack((BetaIFFTprime,BetaIFFTprime1)) if BetaIFFTprime.size else BetaIFFTprime1
                    # need to stack from all parity
                BetaIFFTprime = np.prod(BetaIFFTprime,axis=0) # pointwise product of distribution
            else:
                BetaIFFTprime = np.empty((0,0)).astype(float)
                # Information sections connected to this parity section (assuming no parity over parity sections)
                Indices = np.where(G[i])[0]
                # FFT
                BetaFFTprime = BetaFFT[Indices,:]
                # Multiply the relevant FFTs
                BetaFFTprime = np.prod(BetaFFTprime,axis=0)
                # IFFT
                BetaIFFTprime = np.fft.irfft(BetaFFTprime).real            
            mu[i*M:(i+1)*M] = approximateVector(BetaIFFTprime, K).reshape(-1,1)

    return mu


# In[ ]:


def dynamicDenoiser2(r,OuterCode,K,tau,d,numBPiter):
    """
    Args:
        r (float): Effective observation
        d (float): Signal amplitude
        tau (float): Standard deviation of noise
    """
    M = OuterCode.getsparseseclength()
    L = OuterCode.getvarcount()

    p0 = 1-(1-1/M)**K
    p1 = p0*np.ones(r.shape,dtype=float)
    mu = np.zeros(r.shape,dtype=float)

    # Compute local estimate (lambda) based on effective observation using PME.
    localEstimates = pme0(p0, r, d, tau)
    
    # Reshape local estimate (lambda) into an LxM matrix
    Beta = localEstimates.reshape(L,-1)
    OuterCode.reset()
    for varnodeid in OuterCode.getvarlist():
        i = varnodeid - 1
        Beta[i,:] = approximateVector(Beta[i,:], K)
        OuterCode.setobservation(varnodeid, Beta[i,:]) # CHECK
    
    for iter in range(1):    # CHECK: Leave at 1 for now
        OuterCode.updatechecks()
        OuterCode.updatevars()

    for varnodeid in OuterCode.getvarlist():
        i = varnodeid - 1
#         Beta[i,:] = OuterCode.getestimate(varnodeid)
        Beta[i,:] = OuterCode.getextrinsicestimate(varnodeid)
        mu[i*M:(i+1)*M] = approximateVector(Beta[i,:], K).reshape(-1,1)

    return mu


# In[ ]:


def dynamicDenoiser3(r,G,messageBlocks,L,M,K,tau,d,numBPiter):
    """
    Args:
        r (float): Effective observation
        d (float): Signal amplitude
        tau (float): Standard deviation of noise
    """
    p0 = 1-(1-1/M)**K
    p1 = p0*np.ones(r.shape,dtype=float)
    mu = np.zeros(r.shape,dtype=float)

    # Compute local estimate (lambda) based on effective observation using PME.
    localEstimates = pme0(p0, r, d, tau)
    
    # Reshape local estimate (lambda) into an LxM matrix
    Beta = localEstimates.reshape(L,-1)
    for i in range(L):
        Beta[i,:] = approximateVector(Beta[i,:], K)

        # Rotate PME 180deg about y-axis
        Betaflipped = np.hstack((Beta[:,0].reshape(-1,1),np.flip(Beta[:,1:],axis=1)))
        # Compute and store all FFTs
        BetaFFT = np.fft.rfft(Beta)
        BetaflippedFFT = np.fft.rfft(Betaflipped)
        for i in range(L):
            if messageBlocks[i]:
                # Parity sections connected to info section i
                parityIndices = np.where(G[i])[0]   # Identities of parity block(s) attached
                BetaIFFTprime = np.empty((0,0)).astype(float)
                for j in parityIndices:  # Compute message for check associated with parity j
                    # Other info blocks connected to this parity block
                    messageIndices = np.setdiff1d(np.where(G[j])[0],i)  ## all indicies attahced to j, other than i
                    BetaFFTprime = np.vstack((BetaflippedFFT[j],BetaflippedFFT[messageIndices,:]))  ## j is not part of G[j]
                    # Multiply the relevant FFTs
                    BetaFFTprime = np.prod(BetaFFTprime,axis=0)
                    # IFFT
                    BetaIFFTprime1 = np.fft.irfft(BetaFFTprime).real # multiple parity
                    BetaIFFTprime = np.vstack((BetaIFFTprime,BetaIFFTprime1)) if BetaIFFTprime.size else BetaIFFTprime1
                    # need to stack from all parity
                BetaIFFTprime = np.prod(BetaIFFTprime,axis=0) # pointwise product of distribution
            else:
                BetaIFFTprime = np.empty((0,0)).astype(float)
                # Information sections connected to this parity section (assuming no parity over parity sections)
                Indices = np.where(G[i])[0]
                # FFT
                BetaFFTprime = BetaflippedFFT[Indices,:]
                # Multiply the relevant FFTs
                BetaFFTprime = np.prod(BetaFFTprime,axis=0)
                # IFFT
                BetaIFFTprime = np.fft.irfft(BetaFFTprime).real
            mu[i*M:(i+1)*M] = approximateVector(BetaIFFTprime, K).reshape(-1,1)

    return mu


# ## AMP
# This is the actual AMP algorithm. It's a mostly straightforward transcription from the relevant equations, but note we use `longdouble` types because the expentials are often too big to fit into a normal `double`.

# In[ ]:


def amp_state_update(z, s, P, L, M, Ab, Az, K, G, messageBlocks, denoiserType, numBPiter,OuterCode):

    """
    Args:
        s: State update through AMP composite iteration
        z: Residual update through AMP composite iteration
        tau (float): Standard deviation of noise
        mu: Product of messages from adjoining factors
    """
    n = z.size
    d = np.sqrt(n*P/L)

    # Compute tau online using the residual
    tau = np.sqrt(np.sum(z**2)/n)

    # Compute effective observation
    r = (d*s + Az(z))

    # Compute updated state
    # HERE: It remains unclear what to constrain and renormalize
    if denoiserType==0:
        # Use the uninformative prior p0 for Giuseppe's scheme
        p0 = 1-(1-1/M)**K
        s = pme0(p0, r, d, tau)
    elif denoiserType==1:
        mu = dynamicDenoiser1(r,G,messageBlocks,L,M,K,tau,d,numBPiter)
        s = pme0(mu, r, d, tau)
    else:
        mu = dynamicDenoiser2(r,OuterCode,K,tau,d,numBPiter)
        s = pme0(mu, r, d, tau)
        
    return s


# In[ ]:


def amp_residual(y, z, s1, s2, d1, d2, Ab1, Ab2):
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
    
    if d1 == 0:
        Onsager = (d2**2)*(np.sum(s2) - np.sum(s2**2)) 
        yDecoded = d2*Ab2(s2)
    elif d2 == 0:
        Onsager = (d1**2)*(np.sum(s1) - np.sum(s1**2))
        yDecoded = d1*Ab1(s1)
    else:
        Onsager = (d1**2)*(np.sum(s1) - np.sum(s1**2)) + (d2**2)*(np.sum(s2) - np.sum(s2**2))
        yDecoded = d1*Ab1(s1) + d2*Ab2(s2)
    

    # Compute residual
    z_plus = y - yDecoded + (z/(n*tau**2))*(Onsager)
    
    return z_plus

# # Outer Tree decoder
# 
# This function implements the tree deocoder for a specific graph corresponding to the outer tree code
# 
# It is currently hard-coded for a specfic architecture
# 
# The architecture is based on a tri-adic design and can be found in the simulation results section of https://arxiv.org/pdf/2001.03705.pdf

# In[ ]:


def Tree_decoder(cs_decoded_tx_message,G,L,J,B,listSize):
    
    tree_decoded_tx_message = np.empty(shape=(0,0))
    
    Paths012 = merge_pathst2(cs_decoded_tx_message[:,0:3])
    Paths345 = merge_pathst2(cs_decoded_tx_message[:,3:6])    
    Paths678 = merge_pathst2(cs_decoded_tx_message[:,6:9])
    Paths91011 = merge_pathst2(cs_decoded_tx_message[:,9:12])    
    Paths01267812 = merge_pathslevel2t2(Paths012,Paths678,cs_decoded_tx_message[:,[0,6,12]])    
    Paths3459101115 = merge_pathslevel3t2(Paths345,Paths91011,cs_decoded_tx_message[:,[4,10,15]])   
    Paths01267812345910111513 = merge_all_paths0t2(Paths01267812,Paths3459101115,cs_decoded_tx_message[:,[1,9,13]])
    Paths = merge_all_paths_finalt2(Paths01267812345910111513,cs_decoded_tx_message[:,[3,7,14]])
      
    return Paths

def merge_pathst2(A):
    listSize = A.shape[0]
    B = np.array([np.mod(A[:,0] + a,2**16) for a in A[:,1]]).flatten()
     
    Paths=np.empty((0,0))
    
    for i in range(listSize):
        I = np.where(B==A[i,2])[0].reshape(-1,1)
        if I.size:
            I1 = np.hstack([np.mod(I,listSize).reshape(-1,1),np.floor(I/listSize).reshape(-1,1)]).astype(int)
            Paths = np.vstack((Paths,np.hstack([I1,np.repeat(i,I.shape[0]).reshape(-1,1)]))) if Paths.size else np.hstack([I1,np.repeat(i,I.shape[0]).reshape(-1,1)])
    
    return Paths

def merge_pathslevel2t2(Paths012,Paths678,A):
    listSize = A.shape[0]
    Paths0 = Paths012[:,0]
    Paths6 = Paths678[:,0]
    B = np.array([np.mod(A[Paths0,0] + a,2**16) for a in A[Paths6,1]]).flatten()
    
    Paths=np.empty((0,0))
    
    for i in range(listSize):
        I = np.where(B==A[i,2])[0].reshape(-1,1)
        if I.size:
            I1 = np.hstack([np.mod(I,Paths0.shape[0]).reshape(-1,1),np.floor(I/Paths0.shape[0]).reshape(-1,1)]).astype(int)
            PPaths = np.hstack((Paths012[I1[:,0]].reshape(-1,3),Paths678[I1[:,1]].reshape(-1,3),np.repeat(i,I1.shape[0]).reshape(-1,1)))
            Paths = np.vstack((Paths,PPaths)) if Paths.size else PPaths
               
    return Paths

def merge_pathslevel3t2(Paths345,Paths91011,A):
    listSize = A.shape[0]
    Paths4 = Paths345[:,1]
    Paths10 = Paths91011[:,1]
    B = np.array([np.mod(A[Paths4,0] + a,2**16) for a in A[Paths10,1]]).flatten()
    
    Paths=np.empty((0,0))
    
    for i in range(listSize):
        I = np.where(B==A[i,2])[0].reshape(-1,1)
        if I.size:
            I1 = np.hstack([np.mod(I,Paths4.shape[0]).reshape(-1,1),np.floor(I/Paths4.shape[0]).reshape(-1,1)]).astype(int)
            PPaths = np.hstack((Paths345[I1[:,0]].reshape(-1,3),Paths91011[I1[:,1]].reshape(-1,3),np.repeat(i,I1.shape[0]).reshape(-1,1)))
            Paths = np.vstack((Paths,PPaths)) if Paths.size else PPaths
    
    return Paths


def merge_all_paths0t2(Paths01267812,Paths3459101115,A):
    listSize = A.shape[0]
    Paths1 = Paths01267812[:,1]
    Paths9 = Paths3459101115[:,3]

    B = np.array([np.mod(A[Paths1,0] + a,2**16) for a in A[Paths9,1]]).flatten()
    
    Paths=np.empty((0,0))
    
    for i in range(listSize):
        I = np.where(B==A[i,2])[0].reshape(-1,1)
        if I.size:
            I1 = np.hstack([np.mod(I,Paths1.shape[0]).reshape(-1,1),np.floor(I/Paths1.shape[0]).reshape(-1,1)]).astype(int)
            PPaths = np.hstack((Paths01267812[I1[:,0]].reshape(-1,7),Paths3459101115[I1[:,1]].reshape(-1,7),np.repeat(i,I1.shape[0]).reshape(-1,1)))
            Paths = np.vstack((Paths,PPaths)) if Paths.size else PPaths
    
    return Paths

def merge_all_paths_finalt2(Paths01267812345910111513,A):
    
    listSize = A.shape[0]
    Paths3 = Paths01267812345910111513[:,7]
    Paths7 = Paths01267812345910111513[:,4]
    B = np.mod(A[Paths3,0] + A[Paths7,1] ,2**16)
    
    Paths=np.empty((0,0))
    
    for i in range(listSize):
        I = np.where(B==A[i,2])[0].reshape(-1,1)
        if I.size:
            PPaths = np.hstack((Paths01267812345910111513[I].reshape(-1,15),np.repeat(i,I.shape[0]).reshape(-1,1)))
            Paths = np.vstack((Paths,PPaths)) if Paths.size else PPaths
    return Paths


# If tree decoder outputs more than $K$ valid paths, retain $K-\delta$ of them based on their LLRs
# 
# $\delta$ is currently set to zero

# In[ ]:


def pick_topKminusdelta_paths(Paths, cs_decoded_tx_message, s, J,K,delta):
    
    L1 = Paths.shape[0]
    LogL = np.zeros((L1,1))
    for i in range(L1):
        msg_bit=np.empty(shape=(0,0))
        path = Paths[i].reshape(1,-1)
        for j in range(path.shape[1]):
            msg_bit = np.hstack((msg_bit,j*(2**J)+cs_decoded_tx_message[path[0,j],j].reshape(1,-1))) if msg_bit.size else j*(2**J)+cs_decoded_tx_message[path[0,j],j]
            msg_bit=msg_bit.reshape(1,-1)
        LogL[i] = np.sum(np.log(s[msg_bit])) 
    Indices =  LogL.reshape(1,-1).argsort()[0,-(K-delta):]
    Paths = Paths[Indices,:].reshape(((K-delta),-1))
    
    return Paths


# # Simulation

# In[ ]:


K1=25 # Number of active users in group 1
K2=25 # Number of active users in group 2

B1=128 # Payload size of every active user in group 1
B2=96 # Payload size of every active user in group 2

L1=16 # Number of sections/sub-blocks in group 1
L2=16 # Number of sections/sub-blocks in group 2

n=38400 # Total number of channel uses (real d.o.f)
T=16 # Number of AMP iterations

listSize1 = 2*K1  # List size retained for each section after AMP converges
listSize2 = 2*K2

J=16  # Length of each coded sub-block
M=2**J # Length of each section

messageBlocks = np.array([1,1,0,1,1,0,1,1,0,1,1,0,0,0,0,0]).astype(int) # Indicates the indices of information blocks
# Adjacency matrix of the outer code/graph
G = np.zeros((L1,L1)).astype(int)
# G contains info on what parity blocks a message is attached to and what message blocks a parity is involved with
# Currently, we do not allow parity over parities. BP code needs to be modified a little to accomodate parity over parities
G[0,[2,12]]=1
G[1,[2,13]]=1
G[2,[0,1]]=1
G[3,[5,14]]=1
G[4,[5,15]]=1
G[5,[3,4]]=1
G[6,[8,12]]=1
G[7,[8,14]]=1
G[8,[6,7]]=1
G[9,[11,13]]=1
G[10,[11,15]]=1
G[11,[9,10]]=1
G[12,[0,6]]=1
G[13,[1,9]]=1
G[14,[3,7]]=1
G[15,[4,10]]=1
denoiserType = 1 # Select denoiser: 0 - Original PME; 1 - Dynamic PME; 2+ - Natrual BP.
numBPiter = 1; # Number of BP iterations on outer code. 1 seems to be good enough & AMP theory including state evolution valid only for one BP iteration
delta = 0


# In[ ]:


def simulate(EbNodB, case):
    simCount = 100 # number of simulations

    # EbN0 in linear scale
    EbNo = 10**(EbNodB/10)
    P1 = 2*B1*EbNo/n
    P2 = 2*B2*EbNo/n
    σ_n = 1
    #Generate the power allocation and set of tau coefficients

    # We assume equal power allocation for all the sections. Code has to be modified a little to accomodate non-uniform power allocations
    Phat1 = n*P1/L1
    Phat2 = n*P2/L2
    d1 = np.sqrt(n*P1/L1)
    d2 = np.sqrt(n*P2/L2)

    msgDetected1=0
    msgDetected2=0
    totalTime=0

    for simIndex in range(simCount):
        print('Simulation Number: ' + str(simIndex))

        # Generate active users message sequences
        tx_message1 = np.random.randint(2, size=(K1,B1))
        tx_message2 = np.random.randint(2, size=(K2,B2))

        # Outer-encode the message sequences
        encoded_tx_message_indices = Tree_encode(tx_message1,K1,messageBlocks,G,L1,J)
        codewords = OuterCode.encodemessages(tx_message2)

        # Convert indices to sparse representation
        # sTrue: True state
        sTrue1 = convert_indices_to_sparse(encoded_tx_message_indices, L1, J, K1)
        sTrue2 = np.sum(codewords,axis=0).reshape(-1,1)

        # Generate the binned SPARC codebook
        Ab1, Az1 = sparc_codebook(L1, M, n, P1)
        Ab2, Az2 = sparc_codebook(L2, M, n, P2)

        # Generate our transmitted signal X
        x = d1*Ab1(sTrue1) + d2*Ab2(sTrue2)

        # Generate random channel noise and thus also received signal y
        noise = np.random.randn(n, 1) * σ_n
        y = (x + noise).reshape(-1, 1)
        
        tic = time.time()

        z = y.copy()
        z1 = y.copy()
        z2 = y.copy()
        s1 = np.zeros((L1*M, 1))
        s2 = np.zeros((L2*M, 1))
        
        if case==0:
            # Decode group 1 first
            for t in range(T):
                s1 = amp_state_update(z1, s1, P1, L1, M, Ab1, Az1, K1, G, messageBlocks, 1, numBPiter,OuterCode)
                z1 = amp_residual(y, z1, s1, s2, d1, 0, Ab1, Ab2)
                
            # Interference cancellation
            yUpdated = y - Ab1(s1)
            z2 = yUpdated.copy()
            
            # Decode group 2
            for t in range(T):
                s2 = amp_state_update(z2, s2, P2, L2, M, Ab2, Az2, K2, G, messageBlocks, 2, numBPiter,OuterCode)
                z2 = amp_residual(yUpdated, z2, s1, s2, 0, d2, Ab1, Ab2)
            
        elif case==1:
            for t in range(T):
                s1 = amp_state_update(z, s1, P1, L1, M, Ab1, Az1, K1, G, messageBlocks, 1, numBPiter,OuterCode)
                s2 = amp_state_update(z, s2, P2, L2, M, Ab2, Az2, K2, G, messageBlocks, 2, numBPiter,OuterCode)

                z = amp_residual(y, z, s1, s2, d1, d2, Ab1, Ab2)
                
        else:
            for t in range(T):
                s1 = amp_state_update(z1, s1, P1, L1, M, Ab1, Az1, K1, G, messageBlocks, 1, numBPiter,OuterCode)
                s2 = amp_state_update(z2, s2, P2, L2, M, Ab2, Az2, K2, G, messageBlocks, 2, numBPiter,OuterCode)

                z1 = amp_residual(y, z1, s1, s2, d1, 0, Ab1, Ab2)
                z2 = amp_residual(y, z2, s1, s2, 0, d2, Ab1, Ab2)
                    
 
        
        # Convert decoded sparse vector into vector of indices  
        cs_decoded_tx_message1 = convert_sparse_to_indices(s1, L1, J, listSize1)

        # Tree decoder to decode individual messages from lists output by AMP
        Paths1 = Tree_decoder(cs_decoded_tx_message1,G,L1,J,B1,listSize1)

        # Re-align paths to the correct order
        perm1 = np.argsort(np.array([0,1,2,6,7,8,12,3,4,5,9,10,11,15,13,14]))
        Paths1 = Paths1[:,perm1]

        # If tree deocder outputs more than K valid paths, retain only K of them
        if Paths1.shape[0] > K1:
            Paths1 = pick_topKminusdelta_paths(Paths1, cs_decoded_tx_message1, s1, J, K1,0)

        # Extract the message indices from valid paths in the tree    
        Tree_decoded_indices1 = extract_msg_indices(Paths1,cs_decoded_tx_message1, L1, J)

        print('Graph Decode')

        # Decoding wiht Graph
        originallist = codewords.copy()
        recoveredcodewords = FGG.decoder(OuterCode,s2,2*K2)
        
        toc = time.time()
        totalTime = totalTime + toc - tic
        #print(totalTime)


        # Calculation of per-user prob err
        simMsgDetected1 = 0
        simMsgDetected2 = 0
        for i in range(K1):
            simMsgDetected1 = simMsgDetected1 + np.equal(encoded_tx_message_indices[i,:],Tree_decoded_indices1).all(axis=1).any()
        matches = FGG.numbermatches(originallist,recoveredcodewords)

        print('Group 1: ' + str(simMsgDetected1) + ' out of ' + str(K1))
        print('Group 2: ' + str(matches) + ' out of ' + str(K2))
        msgDetected1 = msgDetected1 + simMsgDetected1
        msgDetected2 = msgDetected2 + matches

    errorRate1= (K1*simCount - msgDetected1)/(K1*simCount)
    errorRate2= (K2*simCount - msgDetected2)/(K2*simCount)
    avgTime = totalTime/simCount
    return errorRate1, errorRate2, avgTime


# # Simulate cases

# In[ ]:


# Case 0: SIC
# Case 1: Joint decoding
# Case 2: TIN


numCases = 3
EbN0dB = np.array([1.8, 2.0, 2.2, 2.4, 2.6, 2.8])
errorRate1 = np.zeros((numCases, len(EbN0dB)))
errorRate2 = np.zeros((numCases, len(EbN0dB)))
times = np.zeros((numCases, len(EbN0dB)))

# Notify commencement of simulation
print('***Starting Simulations*****')


for case in range(numCases):
    for idxSnr in range(len(EbN0dB)):
  
        errorRate1[case, idxSnr], errorRate2[case, idxSnr], times[case, idxSnr] = simulate(EbN0dB[idxSnr], case)
        
        print("\n", file=open("result1.txt", "a"))
        print("(case, EbN0dB)=", (case, EbN0dB[idxSnr]), file=open("result1.txt", "a"))
        print("Per user probability of error (Group 1) = ", errorRate1[case, idxSnr], file=open("result1.txt", "a"))
        print("Per user probability of error (Group 2) = ", errorRate2[case, idxSnr], file=open("result1.txt", "a"))
        print("Times:",times[case, idxSnr], file=open("result1.txt", "a"))
        print("\n", file=open("result1.txt", "a"))


# Notify completion
print('*****All Simulations Complete*****')







