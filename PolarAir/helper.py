# -*- coding: utf-8 -*-
"""
Set of functions that are used from other files
1. dec2bin: From a decimal value to a binary numpy array
2. bin2dec: From a binray numpy array to the corresponding decimal value
3. crcEncoder, crcDecoder: set of functions for CRC encoder and Decoder

"""

import numpy as np

# === Convert an array of integers to a 2D array of binary Strings === #
def dec2bin(listDec,nBits):
    
    # From decimal to string(binary) array
    binary_repr_v = np.vectorize(np.binary_repr)
    binaryString = binary_repr_v(listDec,width=nBits)
    
    # Number of integers
    n = listDec.shape
    if(len(n) != 0):
        n = len(listDec)
    else:
        n = 1
        # Split the string
        binStr = list(map(int,binaryString))
  
        # Save the binary stream
        return np.real(binStr)
        


    # Define a 2D array to save the binary streams
    mtxBinary = np.zeros((n,nBits))
    
    # For each string, break the string and make it array int
    for i in range(n):
        # Take the next string
        binStr = binaryString[i]
        
        # Split the string
        binStr = list(map(int,binStr))
  
        # Save the binary stream
        mtxBinary[i] = np.real(binStr)
        
    return mtxBinary



def bin2dec(binArray):
    
    """ Function to convert from binary numpy array to the corresponding decimal
        Inputs:
            1. binArray: A binary array MSB - LSB 
        Outputs:
            1. The corresponding decimal value
    """
    
    # --- Find the length of the binary array, i.e. number of bits
    bits = binArray.shape[0]

    # --- Create a array where its values are the powers of 2, i.e. [0, 2, 4, 8, ... ]
    temp = np.flip(2 ** np.arange(bits))
    
    # --- Convert from binary to decimal    
    return int(np.dot(binArray,temp))


# ========================= CRC Encoder/Decoder ======================= #

# ---- Encoder
def crcEncoder(dividend ,divisor):
    """ Function to compute the reminder of a binary string and append the reminder to the end
        Inputs:
            1. divided: A binary numpy array to be encoded
            2. divisor: CRC polynomial
        Outputs:
            1. CRC codeword
    """
    # --- Find the length of the two binary arrays
    n1, n2 = len(dividend),len(divisor)
    
    # --- Append n2 zeros to dividend ( First step of CRC encoding )
    newDividend = np.append(dividend, np.zeros(n2, dtype = int)).astype(int)
    
    # --- Compute the reminder ( Second step of CRC encoding )
    i = 0
    while(1):
        
        # XOR the first n1 bits
        newDividend[i:i+n2] = newDividend[i:i+n2] ^ divisor

        # Check if you have to shift to the next bits
        while(newDividend[i] == 0 and i <= n1 ):
            i += 1
           
        # End of the polynomial division    
        if(i > n1 ):
            break

    
    # --- Append the remider to the dividend
    return  np.append(dividend, newDividend[n1:n1+n2])


def crcDecoder(dividend,divisor):
    """ Function to check if the CRC constraint is satisfied 
        Inputs:
            1. divided: A binary numpy array
            2. divisor: CRC polynomial
        Outputs:
             1 or 0 if the constraint is satisfied or not, respectively.  
    """
    
    # --- Find the length of the two binary arrays
    n1, n2 = len(dividend),len(divisor)
    
    # --- Make sure that the type of dividend is int
    dividend = dividend.astype(int)
    
    # --- If the dividend is the all zero msg
    if(sum(dividend) == 0):
        return 1
    else: 
        ones = np.nonzero(dividend)
        # --- If the degree of the dividend is less than the degree of the divisor
        # The msg is wrong
        if(ones[0][0] > n1 - n2):
            return 0
        
    # --- If the dividend doesn't belong to one of these special cases, decode    
    if(ones[0][0] > 0):
        dividend = dividend[ones[0][0]::]
    # --- Compute the reminder
    i = 0
    while(1):
        # XOR the first n1 bits
        if(len(dividend[i:i+n2]) != len(divisor)):
            return 0
        dividend[i:i+n2] = dividend[i:i+n2] ^ divisor
        
        if(sum(dividend) == 0):
            return 1
        
        if(dividend[i] == 1 ):
            # --- Don't Shift
            continue
        else:
            # --- Find the next 1
            ones = np.nonzero(dividend)
            # --- Shift
            i = ones[0][0]
            
            if(len(dividend[i::]) < n2 ):
                # Failure
                return  0
            
            
            


def maxk(array,K):
    idx =  (-array).argsort()[:K]
    return array[idx] , idx
                
        
