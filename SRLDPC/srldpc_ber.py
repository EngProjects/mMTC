"""
Sparse Regression LDPC simulation.
"""

import argparse
import numpy as np
from time import time
from os import system
from os.path import join
from pyfht import block_sub_fht
from joblib import Parallel, delayed
from gfldpc import GFLDPC

__author__ = 'Ebert'
__date__ = '1 November 2023'

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
    r = r0.copy().reshape(n, q)
    alpha = np.exp(d*r/tau**2)
    row_norms = np.linalg.norm(alpha, ord=1, axis=1).reshape(-1, 1)
    return alpha / row_norms

def bp_denoiser(r, code, tau, d, keep_graph, num_bp_iter=1):
    """
    BP Denoiser from Sparse Regression LDPC Codes.

        Arguments:
            r (LMx1 ndarray): AMP effective observation
            code (object): LDPC outer code
            tau (float): AMP effective noise variance parameter
            d (float): amplitude scaling
            keep_graph (bool): flag of whether to retain graph messages
            num_bp_iter (int): number of BP iterations to perform

        Returns:
            q (LMx1 ndarray): denoised state estimate
    """
    n = code.N
    q = code.q

    if keep_graph:
        code.reset_observations()
    else:
        code.reset_graph()

    alpha = compute_likelihood_vector(r, tau, d, n, q)
    for i in range(n):
        code.set_observation(i, alpha[i, :])
    code.bp_decoder(num_bp_iter)
    
    return code.get_estimates().reshape(-1, 1)

def amp_state_update(z, s, d, Az, num_bp_iter, keep_graph, code):
    """
    Compute state update within AMP iterate. 

        Arguments:
            z (nx1 ndarray): AMP residual vector
            s (LMx1 ndarray): AMP state estimate
            d (float): amplitude scaling per entry
            Az (function): SPARC encoding function
            num_bp_iter (int): number of BP iterations to perform
            keep_graph (bool): flag of whether to retain graph messages
            code (object): outer LDPC code object

        Returns:
            s_plus (LMx1 ndarray): updated state estimate
    """
    n = z.size
    tau = np.sqrt(np.sum(z**2)/n)
    r = (d*s + Az(z))
    s_plus = bp_denoiser(r, code, tau, d, keep_graph, num_bp_iter)

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

def simulate(ebnodb, 
             sim_dir, 
             num_amp_iter=25, 
             num_final_bp_iter=100, 
             max_sim_count=10000, 
             num_bp_denoiser_iter=1, 
             keep_graph=False):
    """
    Run SR-LDPC simulation at Eb/N0 = ebnodb dB. Store results in sim_dir.

        Arguments:
            ebnodb (float): Eb/N0 in dB
            sim_dir (str): directory in which to store simulation results
            num_amp_iter (int): number of AMP iterations to perform
            num_final_bp_iter (int): number of BP iterations to run after AMP-BP
            max_sim_count (int): max number of MC simulations to run
            num_bp_denoiser_iter (int): number of BP iterations to run per AMP 
                iteration. If this value is -1, the BP-N schedule is used
            keep_graph (bool): flag of whether to keep BP graph between AMP iters
    
        Returns:
            <none>
    """
    
    # Define LDPC code over finite field by specifying the alist filename
    code_folder = './ldpc_graphs/varying_ldpc_rates/'
    code_filename = 'ldpc_n766_k736_gf256.txt'
    code = GFLDPC(join(code_folder, code_filename))

    # Define filename
    filename = f'./{sim_dir}/ebnodb_{ebnodb}_numampiter_{num_amp_iter}_' + \
               f'numfinalbpiter_{num_final_bp_iter}_numbpdenoiseriter_' + \
               f'{num_bp_denoiser_iter}_keepgraph_{keep_graph}_{code_filename[:-4]}.txt'

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
    target_num_bit_errors = k_bin*10

    # Channel noise parameters
    ebno = 10**(ebnodb/10)
    nvar, sigma = 1, 1
    noise_psd = 2 * nvar
    energy_per_bit = ebno * noise_psd
    total_energy = energy_per_bit * k_bin 
    column_energy_scaling = total_energy / n_gf
    d = np.sqrt(column_energy_scaling)

    # Printing options
    should_print = True
    print_frequency = 500
    write_frequency = 500

    # Error-tracking data structures
    ber = 0.0
    bler = 0.0
    num_bit_errors = 0
    num_block_errors = 0
    num_sims = 0
    snr = 0.0

    # Monte-carlo simulation to estimate BER/BLER
    while (num_bit_errors < target_num_bit_errors) and (num_sims < max_sim_count):
        
        # Periodically update data
        if (should_print and (num_sims % print_frequency == 0)):
            print(f'Eb/No: {ebnodb}dB\tNum Sims: {num_sims}\tBER: {ber}\tBLER: {bler}\tTimestamp: {time()}')
        if (should_print and (num_sims % write_frequency == 0)):
            log = open(filename, 'w')
            log.write(f'BER:\t{ber:.4e}\n')
            log.write(f'BLER:\t{bler:.4e}\n')
            log.write(f'Num Bit Errors:\t{num_bit_errors}\n')
            log.write(f'Completed sims:\t{num_sims}\n')
            log.write(f'Empirical SNR:\t{snr}\n')
            log.close()
        
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

        # Generate transmitted signal
        x = d*Ab(sparse_codeword)

        # Send signal through AWGN channel
        z = sigma*np.random.randn(num_chnl_uses, 1)
        y = x + z
        nvarht = np.linalg.norm(z)**2 / num_chnl_uses
        snr = (num_sims*snr + 10*np.log10(np.linalg.norm(x)**2 / (k_bin*2*nvarht))) / (num_sims+1)

        # Prepare for AMP decoding
        z = y.copy()
        s = np.zeros((q*n_gf, 1))

        # AMP decoding
        for idx_amp_iter in range(num_amp_iter):

            # Adjust number of BP iterations
            if num_bp_denoiser_iter == -1: 
                num_bp_iter = idx_amp_iter + 1
            else:
                num_bp_iter = num_bp_denoiser_iter

            # AMP iterate
            s = amp_state_update(z, s, d, Az, num_bp_iter, keep_graph, code)
            z = amp_residual(y, z, s, d, Ab)
            
            # Check stopping conditions
            cdwd_ht = np.array([np.argmax(s[i*q:(i+1)*q]) for i in range(n_gf)])
            if code.check_consistency(cdwd_ht):
                break  

            # Run BP decoder
            if idx_amp_iter == (num_amp_iter - 1):
                tau = np.sqrt(np.sum(z**2)/num_chnl_uses)
                r = (d*s + Az(z))
                alpha = compute_likelihood_vector(r, tau, d, n_gf, q)
                code.reset_graph()
                for i in range(n_gf):
                    code.set_observation(i, alpha[i, :])
                code.bp_decoder(num_final_bp_iter)
                s = code.get_estimates()

        # Make hard decisions on final estimated vector 
        s = s.reshape(n_gf, q)
        codeword_ht = np.argmax(s, axis=1).flatten().astype(int)
        
        # Compute BER/BLER
        update_bler = True
        for i in range(k_gf):
            err_i = codeword[i]^codeword_ht[i]
            binstr = '0'+ bin(err_i)[2:]
            err_i_bits = [int(bit) for bit in binstr]
            num_bit_errors += np.sum(err_i_bits)
            if update_bler and (np.sum(err_i_bits) > 0):
                num_block_errors += 1
                update_bler = False
        
        # Increment error-tracking parameters
        num_sims += 1
        ber = num_bit_errors/(num_sims*k_bin)
        bler = num_block_errors/(num_sims)

    # Record final results.
    print(f'COMPLETED Eb/No: {ebnodb}dB\tBER: {ber}\tBLER: {bler}')
    log = open(filename, 'w')
    log.write(f'BER:\t{ber:.4e}\n')
    log.write(f'BLER:\t{bler:.4e}\n')
    log.write(f'Num Bit Errors:\t{num_bit_errors}\n')
    log.write(f'Completed sims:\t{num_sims}\n')
    log.write(f'Empirical SNR:\t{snr}\n')
    log.write('========== COMPLETED ==========')
    log.close()

    return

if __name__ == '__main__':
    
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_full_ebno_range', type=int, default=1, help='Flag of whether to run over full range of Eb/N0 values')
    parser.add_argument('--num_amp_iter', type=int, default=25, help='Number of AMP iterations to perform')
    parser.add_argument('--num_final_bp_iter', type=int, default=100, help='Number of BP iterations to perform after AMP-BP process terminates')
    parser.add_argument('--max_sim_count', type=int, default=50000, help='Max number of MC simulations to run')
    parser.add_argument('--num_bp_denoiser_iter', type=int, default=1, help='Number of BP iterations to perform per AMP iteraiton. Value of -1 defaults to BP-N')
    parser.add_argument('--keep_graph', type=int, default=1, help='Flag of whether to keep factor graph messages between AMP iterations')
    args = parser.parse_args()

    # Extract CL arguments
    run_full_ebno_range = args.run_full_ebno_range
    num_amp_iter = args.num_amp_iter
    num_final_bp_iter = args.num_final_bp_iter
    max_sim_count = args.max_sim_count
    num_bp_denoiser_iter = args.num_bp_denoiser_iter
    keep_graph = args.keep_graph

    # Define simulation directory
    sim_dir = 'results'
    system(f'mkdir ./{sim_dir} -p')
    
    # Select Eb/N0 range
    if run_full_ebno_range:
        ebno_db_vals = np.arange(1.0, 3.01, 0.25)
    else:
        ebno_db_vals = np.array([2.5])

    # Execute simulation
    tic = time()
    print(f'Starting simulation at time {tic}')
    res = Parallel(n_jobs=-1)(delayed(simulate)(snr, sim_dir, num_amp_iter, num_final_bp_iter, max_sim_count, num_bp_denoiser_iter, keep_graph) for snr in ebno_db_vals)
    toc = time()
    print(f'Simulation complete. Elapsed time: {toc - tic}s')
