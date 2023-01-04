"""
Sparse Regression LDPC simulation.
"""

import numpy as np
from pyfht import block_sub_fht
from os.path import join

import gfldpc

__author__ = 'Ebert'
__date__ = '4 Jan 2023'

def sparc_codebook(n_gf, q, num_channel_uses):
    """
    Create functions for efficiently computing Ax and A^Tz. 

        Arguments:
            n_gf (int): number of sections
            q (int): length of each section
            num_channel_uses (int): number of channel uses (real d.o.f.)

        Returns:
            Ab(b) (function): computes matrix-vector product Ab
            Az(z) (function): computes matrix-vector product A^Tz
    """
    Ax, Ay, _ = block_sub_fht(num_channel_uses, q, n_gf, seed=None, ordering=None)
    def Ab(b):
        return Ax(b).reshape(-1, 1)/np.sqrt(num_channel_uses)
    def Az(z):
        return Ay(z).reshape(-1, 1)/np.sqrt(num_channel_uses) 
    return Ab, Az

def compute_likelihood_vector(r0, tau, d, n, q):
    """
    Compute likelihood vector according to eq (10). 

        Arguments:
            r0 (LMx1 ndarray): AMP effective observation
            tau (float): AMP effective noise variance parameter
            d (float): amplitude scaling
            n (int): number of sections
            q (int): length of each section

        Returns:
            alpha (MxL ndarray): likelihood vectors for all L sections

    """
    r = r0.copy()
    r = r.reshape(q, n, order='F') 
    col_norms = np.linalg.norm(r, ord=2, axis=0)**2
    alpha = np.ones((q, n), dtype=float)*col_norms
    alpha -= 2*r*d
    alpha += d**2
    alpha = np.exp(-alpha/(2*tau**2))
    return alpha / np.linalg.norm(alpha, ord=1, axis=0)

def bp1_denoiser(r, code, tau, d, num_bp_iter=1):
    """
    BP-1 Denoiser (Def 7) from Sparse Regression LDPC Codes.

        Arguments:
            r (LMx1 ndarray): AMP effective observation
            code (object): LDPC outer code
            tau (float): AMP effective noise variance parameter
            d (float): amplitude scaling
            num_bp_iter (int): number of BP iterations to perform

        Returns:
            q (LMx1 ndarray): denoised state estimate
    """
    n = code.N
    q = code.q
    
    code.reset_observations()
    alpha = compute_likelihood_vector(r, tau, d, n, q)
    for i in range(n):
        code.set_observation(i, alpha[:, i].flatten())
    code.bp_decoder(num_bp_iter)
    
    return code.get_estimates().reshape(-1, 1)

def amp_state_update(z, s, d, Az, num_bp_iter, code):
    """
    Compute state update within AMP iterate. 

        Arguments:
            z (nx1 ndarray): AMP residual vector
            s (LMx1 ndarray): AMP state estimate
            d (float): amplitude scaling per entry
            Az (function): SPARC encoding function
            num_denoiser_bp_iter (int): number of BP iterations to perform
            code (object): outer LDPC code object

        Returns:
            s_plus (LMx1 ndarray): updated state estimate
    """
    n = z.size
    tau = np.sqrt(np.sum(z**2)/n)
    r = (d*s + Az(z))
    s_plus = bp1_denoiser(r, code, tau, d, num_bp_iter)

    return s_plus

def amp_residual(y, z, s, d, Ab):
    """
    Compute residual within AMP iterate.

        Arguments: 
            y (nx1 ndarray): received vector
            z (nx1 ndarray): AMP residual vector
            s (LMx1 ndarray): current AMP state estimate
            d (float): amplitude scaling per entry
            Ab (function): SPARC encoding function

        Returns:
            z (nx1 ndarray): updated AMP residual vector
    """
    n = y.size
    tau = np.sqrt(np.sum(z**2)/n)
    onsager_term = (d**2)*(np.sum(s) - np.sum(s**2))
    z_plus = y - d*Ab(s) + (z/(n*tau**2))*onsager_term
    
    return z_plus

if __name__ == '__main__':
    
    # Define LDPC code over finite field by specifying the alist filename
    code_folder = './LDPCGraphs/'
    code_filename = 'ldpc_n751_k736_gf256_lambda08.txt'
    code = gfldpc.GFLDPC(join(code_folder, code_filename))

    # Code parameters
    q = code.q
    bits_per_symbol = int(np.log2(q))
    k_gf = code.K
    n_gf = code.N
    R_ldpc = k_gf / n_gf
    k_bin = k_gf*bits_per_symbol
    num_chnl_uses = 7350
    R_tot = k_bin/num_chnl_uses

    # Decoder parameters
    num_amp_iter = 100
    num_bp_denoiser_iter = 5
    num_final_bp_iter = 100
    max_sim_count = 10000
    target_num_bit_errors = k_bin*2

    # Channel noise parameters
    ebno_db_vals = np.array([2.0, 2.25, 2.5, 2.75])
    ebno_vals = 10**(ebno_db_vals/10)
    pvals = 2*k_bin*ebno_vals/num_chnl_uses
    sigma = 1
    
    # Print out simulation information
    print('='*25 + 'Sparse Regression LDPC Simulation' + '='*25)
    print('Using (%d, %d) LDPC outer code over GF(2^%d)' \
          % (n_gf, k_gf, bits_per_symbol))
    print('LDPC Code Rate: %1.2f; SRLDPC rate: %1.2f; Channel Uses: %d' \
          % (R_ldpc, R_tot, num_chnl_uses))
    print('num_amp_iter: %d; num_bp_denoiser_iter: %d; num_final_bp_iter: %d' \
          % (num_amp_iter, num_bp_denoiser_iter, num_final_bp_iter))
    print('='*83)

    # Error-tracking data structures
    ber = np.zeros(ebno_db_vals.shape)

    # Iterate through all Eb/No values
    for idxsnr in range(len(ebno_vals)):
        print('Testing Eb/No = ' + str(ebno_db_vals[idxsnr]) + 'dB')

        # Reset simulation parameters for current Eb/No value
        num_bit_errors = 0
        num_sims = 0
        phat = num_chnl_uses*pvals[idxsnr]/n_gf
        d = np.sqrt(phat)

        # Monte-carlo simulation to estimate probability of bit error
        while (num_bit_errors < target_num_bit_errors) and (num_sims < max_sim_count):
            if num_sims % 100 == 0:
                print('Simulation Number: ' + str(num_sims))
            
            # Reset SRLDPC outer factor graph
            code.reset_graph()

            # Generate random binary message
            bin_msg = np.random.randint(2, size=k_bin)
            
            # Encode message
            codeword = code.encode(bin_msg)
            assert code.check_consistency(codeword)

            # Generate sparse representation through indexing
            sparse_codeword = np.zeros(q*n_gf)
            sparse_codeword[np.arange(n_gf)*q+codeword] = 1        

            # Generate the binned SPARC codebook
            Ab, Az = sparc_codebook(n_gf, q, num_chnl_uses)

            # Generate our transmitted signal X
            x = np.sqrt(phat)*Ab(sparse_codeword)

            # Send signal through AWGN channel
            y = x + (sigma*np.random.randn(num_chnl_uses, 1))

            # Prepare for AMP decoding
            z = y.copy()
            s = np.zeros((q*n_gf, 1))

            # AMP decoding
            for t in range(num_amp_iter):
                s = amp_state_update(z, s, d, Az, num_bp_denoiser_iter, code)
                z = amp_residual(y, z, s, d, Ab)
                
                # Check stopping conditions
                cdwd_ht = np.array([np.argmax(s[i*q:(i+1)*q]) for i in range(n_gf)])
                if code.check_consistency(cdwd_ht):
                    break  
                # Run BP decoder
                elif t == (num_amp_iter - 1):
                    tau = np.sqrt(np.sum(z**2)/num_chnl_uses)
                    alpha = compute_likelihood_vector(s, tau, d, n_gf, q)
                    code.reset_graph()
                    for i in range(n_gf):
                        code.set_observation(i, alpha[:, i].flatten())
                    code.bp_decoder(num_final_bp_iter)
                    s = code.get_estimates()
                
            # Make hard decisions on final estimated vector 
            s = s.reshape(q, n_gf, order='F')
            codeword_ht = np.argmax(s, axis=0).astype(int)
            
            # Compute BER
            for i in range(k_gf):
                err_i = codeword[i]^codeword_ht[i]
                binstr = '0'+ bin(err_i)[2:]
                err_i_bits = [int(bit) for bit in binstr]
                num_bit_errors += np.sum(err_i_bits)
                
            if num_sims % 100 == 0:
                print('Cumulative BER: %1.4e' % (num_bit_errors/((num_sims+1)*k_bin)))
            
            # Increment num_sims
            num_sims += 1
        
        # Update BER
        ber[idxsnr] = num_bit_errors/(num_sims*k_bin)
        print('BER for Eb/No = %1.2fdB: %1.4e' % (ebno_db_vals[idxsnr], ber[idxsnr]))

    # Print results:
    print('Eb/N0 (dB) vals: ' + str(ebno_db_vals))
    print('Coded BER, including errors in parity bits: ' + str(ber))
    
