""" SR-LDPC state evolution """

import numpy as np
from time import time
from os.path import join
from pyfht import block_sub_fht
from gfldpc import GFLDPC
from gfldpc_se import GFLDPCSE
from joblib import Parallel, delayed

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

def approximate_ea0(q, N, d, tau2):
    """
    Approximate E[||alpha||^2] via MC simulations

        Arguments:
            q (int): field size
            N (int): number of sections
            tau2 (float): estimate of noise variance

        Returns:
            exp_alpha0 (float): E
    """
    num_trials = 250
    estimate = 0.0
    s = np.zeros(N*q)
    s[np.arange(N)*q] = d
    for idx_trial in range(num_trials):
        r = s + np.sqrt(tau2)*np.random.randn(N*q)
        alpha = compute_likelihood_vector(r, np.sqrt(tau2), d, N, q)
        estimate += 1/(idx_trial+1)*(np.average(alpha[:, 0]) - estimate)

    return estimate

def bp1_denoiser(r, code, tau, d, keep_graph, num_bp_iter=1):
    """
    BP-1 Denoiser (Def 7) from Sparse Regression LDPC Codes.

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
        code.set_observation(i, alpha[i, :].flatten())
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
            num_denoiser_bp_iter (int): number of BP iterations to perform
            keep_graph (bool): flag of whether to retain graph messages
            code (object): outer LDPC code object

        Returns:
            s_plus (LMx1 ndarray): updated state estimate
    """
    n = z.size
    tau = np.sqrt(np.sum(z**2)/n)
    r = (d*s + Az(z))
    s_plus = bp1_denoiser(r, code, tau, d, keep_graph, num_bp_iter)

    return s_plus, tau

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

def density_evolution(q, N, d, tau2, codese, num_bp_iter, keep_graph):
    """
    Track MSE through outer LDPC code. (Note: not true density evolution)

        Arguments:
            q (int): field size
            N (int): length of LDPC code
            d (float): amplitude scaling
            tau2 (float): estimate of noise variance
            codese (object): outer LDPC DE object
            num_bp_iter (int): number of BP iterations to perform
            keep_graph (bool): flag of whether to retain graph messages

        Returns:
            mse (float): output MSE of BP algorithm
    """

    if keep_graph:
        codese.reset_observations()
    else:
        codese.reset_graph()
    
    ea0 = approximate_ea0(q, N, d, tau2)

    for i in range(N):
        codese.set_observation(i, ea0)
    
    codese.density_evolution(num_bp_iter)

    return codese.get_mse()

def simulate(ebnodb, num_bp_denoiser_iter, N):
    """
    Monte-carlo simulation of MSE

        Arguments:
            ebnodb (float): Eb/N0 in dB
            num_bp_denoiser_iter (int): number of BP iterations to perform per AMP iteration
            N (int): length of LDPC code

        Returns:
            <none>
    """
    
    # Define LDPC code over finite field by specifying the alist filename
    code_folder = './ldpc_graphs/varying_ldpc_rates/'
    code_filename = f'ldpc_n{N}_k736_gf256.txt'
    alist_path = join(code_folder, code_filename)
    code = GFLDPC(alist_path)
    codese = GFLDPCSE(alist_path)

    # Code parameters
    q = code.q
    bits_per_symbol = int(np.log2(q))
    k_gf = code.K
    n_gf = code.N
    R_ldpc = k_gf / n_gf
    k_bin = k_gf*bits_per_symbol
    num_chnl_uses = 7350
    R_tot = k_bin/num_chnl_uses
    assert N == n_gf

    # Denoiser parameters
    # num_bp_denoiser_iter = -1 results in BP-N schedule
    # keep_graph_messages results in denoiser only resetting local observations
    keep_graph =  True

    # Decoder parameters
    num_amp_iter = 20
    max_sim_count = 500

    # Channel noise parameters
    ebno = 10**(ebnodb/10)
    column_energy_scaling = 1
    d = np.sqrt(column_energy_scaling)
    total_energy = n_gf*d**2
    energy_per_bit = total_energy / k_bin
    nvar = energy_per_bit / (2*ebno)
    sigma = np.sqrt(nvar)

    # Error-tracking data structures
    tau2s = np.zeros(num_amp_iter)
    tau2s_ht = np.zeros(num_amp_iter)
    tau2s_ht[0] = (n_gf*d**2)/num_chnl_uses + nvar

    # Iterate through all Eb/No values
    for sim_num in range(max_sim_count):
            
        # Reset SRLDPC outer factor graph
        code.reset_graph()
        codese.reset_graph()

        # Generate sparse representation through indexing - use all zero codeword
        sparse_codeword = np.zeros(q*n_gf)
        sparse_codeword[np.arange(n_gf)*q] = 1        

        # Generate the binned SPARC codebook
        Ab, Az = sparc_codebook(n_gf, q, num_chnl_uses)

        # Generate our transmitted signal X
        x = d*Ab(sparse_codeword)

        # Send signal through AWGN channel
        z = sigma*np.random.randn(num_chnl_uses, 1)
        y = x + z 

        # Prepare for AMP decoding
        z = y.copy()
        s = np.zeros((q*n_gf, 1))

        # AMP decoding
        for idx_amp_iter in range(num_amp_iter):

            if num_bp_denoiser_iter == -1:
                num_bp_iter = idx_amp_iter + 1
            else:
                num_bp_iter = num_bp_denoiser_iter

            s, tau = amp_state_update(z, s, d, Az, num_bp_iter, keep_graph, code)
            z = amp_residual(y, z, s, d, Ab)

            tau2s[idx_amp_iter] += tau**2 / max_sim_count

            if sim_num == 0:
                if num_bp_iter == 0:
                    mse = (1 - approximate_ea0(q, n_gf, d, tau2s_ht[idx_amp_iter]))*n_gf
                else:
                    mse = density_evolution(q, n_gf, d, tau2s_ht[idx_amp_iter], codese, num_bp_iter, keep_graph)

                # theoretic_mses[idx_amp_iter] = mse / (n_gf*q)
                if idx_amp_iter < num_amp_iter-1:
                    tau2s_ht[idx_amp_iter+1] = mse/num_chnl_uses + nvar

    # Save results
    np.savetxt(f'tau2s_N_{N}_{ebnodb}_{num_bp_denoiser_iter}.txt', tau2s)
    np.savetxt(f'tau2_hts_N_{N}_{ebnodb}_{num_bp_denoiser_iter}.txt', tau2s_ht)

    return nvar

if __name__ == '__main__':

    num_bp_denoiser_iters = np.array([1])
    snrs = np.array([2.5])
    Nvals = np.array([1100, 1080, 1060, 1040, 1020, 1000, 980, 960, 940, 920, 898, 876, 856, 836, 818, 800, 783, 766, 751, 739])

    tic = time()
    nvars = Parallel(n_jobs=-1)(delayed(simulate)(ebnodb, numbpiter, N) for ebnodb in snrs for numbpiter in num_bp_denoiser_iters for N in Nvals)
    toc = time()

    print(nvars)
