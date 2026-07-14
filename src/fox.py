import math, sys
import numpy as np 
from numba import njit
from scipy.optimize import linear_sum_assignment

@njit(fastmath=False, error_model='numpy')
def ibp_sample_prior(d, alpha, max_features=1000, enforce_nonempty_rows=True):
    Z = np.zeros((d, max_features), dtype=np.int32)
    m = np.zeros(max_features, dtype=np.int32)

    K = np.random.poisson(alpha)

    if K == 0:
        K = 1
    if K > max_features:
        K = max_features

    for k in range(K):
        Z[0, k] = 1
        m[k] = 1

    for i in range(1, d):
        row_sum = 0

        for k in range(K):
            if np.random.rand() < m[k] / (i + 1):
                Z[i, k] = 1
                m[k] += 1
                row_sum += 1

        # New dishes
        new_K = np.random.poisson(alpha / (i + 1))

        for _ in range(new_K):
            if K < max_features:
                Z[i, K] = 1
                m[K] = 1
                K += 1
                row_sum += 1
            else:
                break

        # Optional: enforce at least one feature per row
        if enforce_nonempty_rows and row_sum == 0:
            if K < max_features:
                Z[i, K] = 1
                m[K] = 1
                K += 1
            else:
                # fallback: assign an existing feature
                k = np.random.randint(0, K)
                Z[i, k] = 1
                m[k] += 1

    return Z[:, :K]

@njit(fastmath=False, error_model='numpy')
def likelihood(y, x, phi, sigma2):
    return np.exp(-0.5 * (y-phi*x)*(y-phi*x) / sigma2 ) / np.sqrt(2*np.pi*sigma2)

@njit(fastmath=False, error_model='numpy')
def forward_likelihood(y, x, etas, f, phi, sigma2):

    T = len(y)  

    if np.sum(f) == 0:
        return -np.inf

    eta_f = etas * f
    pi = eta_f / np.sum(eta_f, axis=1)[:, None]

    # Uniform initial distribution
    alpha = f / np.sum(f)

    # Log-likelihood accumulator
    loglik = 0.0
    
    for t in range(T):

        # Emission probabilities p(y_t | z_t = k)
        emission = likelihood(y[t], x[t], phi, sigma2)

        # Multiply by emission
        alpha *= emission

        # Normalisation (scaling)
        c = np.sum(alpha)
        loglik += np.log(c)
        alpha = alpha / c

        # Propagate forward (except last step)
        alpha = alpha @ pi

    return loglik

@njit(fastmath=False, error_model='numpy')
def sample_categorical(probs):
    u = np.random.rand()
    cum = 0.0
    for k in range(len(probs)):
        cum += probs[k]
        if u < cum:
            return k
    return len(probs) - 1

@njit(fastmath=False, error_model='numpy')
def sample_states(y, x, etas, f, phi, sigma2):

    T = len(y)
    K = len(phi)

    eta_f = etas * f
    pi = eta_f / np.sum(eta_f, axis=1)[:, None]

    alpha = f / np.sum(f)

    filtered = np.zeros((T, K))

    for t in range(T):

        # emission
        diff = y[t] - phi * x[t]
        emission = np.exp(-0.5 * diff * diff / sigma2) / np.sqrt(2*np.pi*sigma2)

        # update
        alpha *= emission
        alpha /= np.sum(alpha)

        filtered[t] = alpha

        # propagate
        alpha = alpha @ pi

    # --- Backward sampling ---
    z = np.zeros(T, dtype=np.int64)

    # sample final state
    z[T-1] = sample_categorical(filtered[T-1])

    # backward recursion
    for t in range(T-2, -1, -1):

        probs = filtered[t] * pi[:, z[t+1]]
        probs /= np.sum(probs)

        z[t] = sample_categorical(probs)

    return z

@njit(fastmath=False, error_model='numpy')
def sample_phi_sigma2_post(sum_yy, sum_xy, sum_xx, n, params):
    lam = params[0]
    a = params[1]
    b = params[2]
    
    a_n = a + 0.5*n
    lam_n = lam + sum_xx
    m_n = sum_xy / lam_n
    b_n = b + 0.5*(sum_yy - m_n*m_n * lam_n)
    
    sigma2 = 1.0/np.random.gamma(a_n, scale=1/b_n)
    phi = np.random.normal(m_n, np.sqrt(sigma2/lam_n))
    return phi, sigma2

@njit(fastmath=False, error_model='numpy')
def get_summary_states_for_state(y, x, states, k):
    
    d, T_ = y.shape
    sum_yy, sum_xy, sum_xx, n = 0.0, 0.0, 0.0, 0.0

    for j in range(d):
        for t in range(T_):
            if states[j, t] == k:
                sum_yy += y[j, t]*y[j, t]
                sum_xy += y[j, t]*x[j, t]
                sum_xx += x[j, t]*x[j, t]
                n += 1.0
    
    return sum_yy, sum_xy, sum_xx, n

@njit(fastmath=False, error_model='numpy')
def get_transition_counts(states, K):

    counts = np.zeros((K, K), dtype=np.float64)

    for i in range(1, len(states)):
        counts[states[i-1], states[i]] += 1
    
    return counts

@njit(fastmath=False, error_model='numpy')
def remove_col(arr, idx):
    if arr.ndim == 1:
        n = arr.shape[0]
        result = np.empty(n - 1, dtype=arr.dtype)
        new_i = 0
        for i in range(n):
            if i != idx:
                result[new_i] = arr[i]
                new_i += 1
        return result
    elif arr.ndim == 2:
        n_rows, n_cols = arr.shape
        result = np.empty((n_rows, n_cols - 1), dtype=arr.dtype)
        for i in range(n_rows):
            new_col = 0
            for j in range(n_cols):
                if j != idx:
                    result[i, new_col] = arr[i, j]
                    new_col += 1
        return result
    else:
        raise ValueError("Only 1D and 2D arrays are supported")

@njit(fastmath=False, error_model='numpy')
def remove_row(arr, row_idx):
    if arr.ndim != 2:
        raise ValueError("Input must be a 2D array")
    
    n_rows, n_cols = arr.shape
    result = np.empty((n_rows - 1, n_cols), dtype=arr.dtype)
    
    new_row = 0
    for i in range(n_rows):
        if i != row_idx:
            for j in range(n_cols):
                result[new_row, j] = arr[i, j]
            new_row += 1
            
    return result

@njit(fastmath=False, error_model='numpy')
def add_zero_element_or_col(arr, add_row=False):
    if arr.ndim == 1:
        n = arr.shape[0]
        new_arr = np.zeros(n + 1, dtype=arr.dtype)
        new_arr[:n] = arr
        return new_arr

    elif arr.ndim == 2:
        n_rows, n_cols = arr.shape
        if add_row:
            # add both a row and a column
            new_arr = np.zeros((n_rows + 1, n_cols + 1), dtype=arr.dtype)
            new_arr[:n_rows, :n_cols] = arr
        else:
            # add a column of zeros
            new_arr = np.zeros((n_rows, n_cols + 1), dtype=arr.dtype)
            new_arr[:, :n_cols] = arr
        return new_arr

    else:
        raise ValueError("Only 1D and 2D arrays are supported")

@njit(fastmath=False, error_model='numpy')
def ni_prob(curr_ni, prev_ni):
    if prev_ni == 0:
        return 1
    else:
        if curr_ni > prev_ni:
            return 0.5
        else:
            return 0.5 / prev_ni

@njit(fastmath=False, error_model='numpy')
def log_dpois(x, lam):
    if x < 0:
        return -np.inf  # log(0)
    
    # Compute log(x!) iteratively
    log_fact = 0.0
    for i in range(1, x + 1):
        log_fact += np.log(i)
    
    return x * np.log(lam) - lam - log_fact

@njit(fastmath=False, error_model='numpy')
def sample_features(y, x, features, phis, sigma2s, etas, alpha, gamma, kappa, params):
    
    d, K = features.shape
    counts = np.sum(features, axis=0)

    for j in range(d):
        
        # update current features
        for k in range(K):
            if features[j, k] == 1:
                counts[k] -= 1

            if counts[k] == 0:
                features[j, k] = 0
                continue
            
            log_p1 = np.log(counts[k] / d) 
            log_p0 = np.log(1 - counts[k] / d)

            features[j, k] = 1
            log_p1 += forward_likelihood(y[j, :], x[j, :], etas[j], features[j, :], phis, sigma2s)
            features[j, k] = 0
            log_p0 += forward_likelihood(y[j, :], x[j, :], etas[j], features[j, :], phis, sigma2s)

            m = max(log_p1, log_p0)
            log_p1 -= m 
            log_p0 -= m 

            log_p1 += 300 
            log_p0 += 300

            p1 = np.exp(log_p1) / (np.exp(log_p1) + np.exp(log_p0))

            if np.random.uniform(0, 1) < p1:
                counts[k] += 1
                features[j, k] = 1

        # clean up features (remove columns k with count[k] == 0)
        for k in range(K-1, -1, -1):
            if counts[k] == 0:
                features = remove_col(features, k)
                counts = remove_col(counts, k)
                for j1 in range(d):                    
                    etas[j1] = remove_col(etas[j1], k)
                    etas[j1] = remove_row(etas[j1], k)
                phis = remove_col(phis, k)
                sigma2s = remove_col(sigma2s, k)
        
        K = features.shape[1]
        
        # now do RJMCMC to add potential unique features for series j
        curr_etas = etas
        curr_phi = phis
        curr_s2 = sigma2s

        curr_features_j = features[j, :]
        curr_ni = 0
        for th in range(1):
            
            if (curr_ni == 0) or (np.random.uniform(0, 1) < 0.5):
                # add new unique feature
                prop_ni = curr_ni + 1
                prop_features_j = add_zero_element_or_col(curr_features_j)
                prop_features_j[-1] = 1

                prop_phi = add_zero_element_or_col(curr_phi)
                prop_s2 = add_zero_element_or_col(curr_s2)
                new_phi, new_s2 = sample_phi_sigma2_post(0, 0, 0, 0, params)
                prop_phi[-1] = new_phi 
                prop_s2[-1] = new_s2
                
                prop_etas = curr_etas.copy()
                for j1 in range(d):
                    prop_etas[j1] = add_zero_element_or_col(prop_etas[j1], add_row=True)
                    for k1 in range(prop_etas[j1].shape[1]):
                        for k2 in range(prop_etas[j1].shape[1]):
                            if k1 == k2:
                                prop_etas[j1][k1, k2] = np.random.gamma(gamma + kappa, 1)
                            else:
                                prop_etas[j1][k1, k2] = np.random.gamma(gamma, 1)

            else:
                # delete
                prop_ni = curr_ni - 1
                del_ind = np.random.randint(0, curr_ni)

                prop_features_j = remove_col(curr_features_j, len(curr_features_j) - del_ind - 1)
                prop_phi = remove_col(curr_phi, len(curr_features_j) - del_ind - 1)
                prop_s2 = remove_col(curr_s2, len(curr_features_j) - del_ind - 1)
                prop_etas = curr_etas.copy()
                for j1 in range(d):
                    prop_etas[j1] = remove_col(prop_etas[j1], len(curr_features_j) - del_ind - 1)
                    prop_etas[j1] = remove_row(prop_etas[j1], len(curr_features_j) - del_ind - 1)
                
            log_accept_prob = np.log(ni_prob(prop_ni, curr_ni)) - np.log(ni_prob(curr_ni, prop_ni))
            log_accept_prob += (log_dpois(curr_ni, alpha/d) - log_dpois(prop_ni, alpha/d))
            log_accept_prob += forward_likelihood(y[j, :], x[j, :], prop_etas[j], prop_features_j, prop_phi, prop_s2)
            log_accept_prob -= forward_likelihood(y[j, :], x[j, :], curr_etas[j], curr_features_j, curr_phi, curr_s2)

            if np.random.uniform(0, 1) < np.exp(log_accept_prob):
                curr_etas = prop_etas
                curr_phi = prop_phi
                curr_s2 = prop_s2

                curr_features_j = prop_features_j
                curr_ni = prop_ni

        for _ in range(len(curr_features_j) - features.shape[1]):
            features = add_zero_element_or_col(features)
            counts = add_zero_element_or_col(counts)
            counts[-1] = 1
        
        etas = curr_etas
        phis = curr_phi
        sigma2s = curr_s2
        features[j, :] = curr_features_j
        K = features.shape[1]

    return features, phis, sigma2s, etas

@njit(fastmath=False, error_model='numpy')
def flatten(arr):
    out = []
    for j in range(arr.shape[0]):
        for t in range(arr.shape[1]):
            out.append(arr[j, t])
    return out

@njit(fastmath=False, error_model='numpy')
def remove_zero_cols_rows(arr, f):

    to_remove = []
    for i in range(len(f)):
        if f[i] == 0:
            to_remove.append(i)
    
    for i in range(len(to_remove)-1, -1, -1):
        arr = remove_row(arr, to_remove[i])
        arr = remove_col(arr, to_remove[i])
    
    return arr

@njit(fastmath=False, error_model='numpy')
def f_gamma(gamma, kappa, etas, F):
    log_val = 0.0
    d = F.shape[0]
    for i in range(d):
        K_i = int(np.sum(F[i, :]))
        eta_f = etas[i] * F[i, :]
        pi = eta_f / np.sum(eta_f, axis=1)[:, None]
        pi = remove_zero_cols_rows(pi, F[i, :])
        for k in range(K_i):
            log_val += math.lgamma(gamma*K_i+kappa) - ((K_i-1)*math.lgamma(gamma) + math.lgamma(gamma+kappa))
            for j in range(K_i):
                if (k == j):
                    log_val += (gamma+kappa-1)*math.log(pi[k, j])
                else:
                    log_val += (gamma-1)*math.log(pi[k, j])

    return log_val

@njit(fastmath=False, error_model='numpy')
def update_gamma(curr_gamma, kappa, etas, F, a, b, prop_sd):
    
    prop_gamma = np.random.gamma((curr_gamma*curr_gamma) / (prop_sd*prop_sd), 1.0/(curr_gamma / (prop_sd*prop_sd)))
    
    theta = (curr_gamma*curr_gamma) / (prop_sd*prop_sd)
    theta_prop = (prop_gamma*prop_gamma) / (prop_sd*prop_sd)

    log_acc_prob = f_gamma(prop_gamma, kappa, etas, F) - f_gamma(curr_gamma, kappa, etas, F)
    log_acc_prob += (math.lgamma(theta) - math.lgamma(theta_prop))
    log_acc_prob += ((theta_prop-theta-a)*math.log(curr_gamma) - (theta-theta_prop-a)*math.log(prop_gamma))
    log_acc_prob += (theta-theta_prop)*math.log(prop_sd*prop_sd) - (prop_gamma-curr_gamma)*b

    if np.random.uniform(0, 1) < np.exp(log_acc_prob):
        return prop_gamma 
    else:
        return curr_gamma

@njit(fastmath=False, error_model='numpy')
def f_kappa(kappa, gamma, etas, F):
    log_val = 0.0
    d = F.shape[0]
    for i in range(d):
        K_i = int(np.sum(F[i, :]))
        eta_f = etas[i] * F[i, :]
        pi = eta_f / np.sum(eta_f, axis=1)[:, None]
        pi = remove_zero_cols_rows(pi, F[i, :])
        log_val += (K_i*math.lgamma(gamma*K_i + kappa) - K_i*math.lgamma(gamma + kappa))
        for j in range(K_i):
            log_val += (gamma+kappa-1)*math.log(pi[j, j])
    return log_val

@njit(fastmath=False, error_model='numpy')
def update_kappa(curr_kappa, gamma, etas, F, a, b, prop_sd):
    prop_kappa = np.random.gamma((curr_kappa*curr_kappa) / (prop_sd*prop_sd), 1.0/(curr_kappa / (prop_sd*prop_sd)))
    
    theta = curr_kappa*curr_kappa / (prop_sd*prop_sd)
    theta_prop = prop_kappa*prop_kappa / (prop_sd*prop_sd)

    log_acc_prob = f_kappa(prop_kappa, gamma, etas, F) - f_kappa(curr_kappa, gamma, etas, F)
    log_acc_prob += (math.lgamma(theta) - math.lgamma(theta_prop))
    log_acc_prob += ((theta_prop-theta-a)*math.log(curr_kappa) - (theta-theta_prop-a)*math.log(prop_kappa))
    log_acc_prob += (theta-theta_prop)*math.log(prop_sd*prop_sd) - (prop_kappa-curr_kappa)*b
    
    if np.random.uniform(0, 1) < np.exp(log_acc_prob):
        return prop_kappa
    else:
        return curr_kappa

@njit(fastmath=False, error_model='numpy')
def predictive_density(y, x, states, phis, sigma2s, etas, F):
    y_pred_est = 1.0
    d, T_ = states.shape
    # print("")
    for j in range(d):
        k = states[j, T_-1]
        pi_k = etas[j][k, :] * F[j, :]
        pi_k /= np.sum(pi_k)

        est = np.sum(likelihood(y[j], x[j], phis, sigma2s) * pi_k)
        # print(pi_k[k])
        # print(est)
        y_pred_est *= est 
    # if y_pred_est > 0.0001:
    #     print(0)
    return y_pred_est

# @njit(fastmath=False, error_model='numpy')
# def predictive_density(y, x, states, phis, sigma2s, etas, F):
#     y_pred_est = 0.0
#     d, T_ = states.shape
#     for j in range(d):
#         k = states[j, T_-1]
#         pi_k = etas[j][k, :] * F[j, :]
#         pi_k /= np.sum(pi_k)
#         new_k = sample_categorical(pi_k)
#         diff = y[j] - phis[new_k]*x[j]
#         y_pred_est += diff*diff / np.sqrt(sigma2s[new_k])
#         # print(pi_k)
#         # print('self transition prob: ', pi_k[k])
#         #est = likelihood(y[j], x[j], phis[new_k], sigma2s[new_k])
#         # est = np.sum(likelihood(y[j], x[j], phis, sigma2s) * pi_k)
#         # y_pred_est *= est 

#     return y_pred_est

@njit(fastmath=False, error_model='numpy')
def sample(y: np.array, 
           n_iter: int,
           burn_in: int,
           y_pred: np.array,
           params: np.array,
           alpha_a: float=1.0, 
           alpha_b: float=1.0, 
           gamma_a: float=1.0, 
           gamma_b: float=1.0, 
           kappa_a: float=100.0, 
           kappa_b: float=1.0, 
           gamma_prop_sd: float = 0.2, 
           kappa_prop_sd: float = 10.0, 
           thin: int=10,
           store_clusters: bool=False,
           verbose: bool = False):
    
    cluster_samples = []

    d = y.shape[0]
    T_ = y.shape[1]

    x = y.copy()
    # add auxiliary y
    y = np.append(y[:, 1:], np.zeros((d, 1), dtype=np.float64), axis=1)
    
    # d, T_ = y.shape

    # initialise variables to prior means
    alpha = alpha_a / alpha_b # IBP alpha prior
    gamma = gamma_a / gamma_b # eta prior gamma
    kappa = kappa_a / kappa_b # eta prior kappa (stickiness parameter)

    features = np.ones((d, 8), dtype=np.int64)

    K = features.shape[1]

    phis = np.random.uniform(0, 1, size=K) # phis
    sigma2s = np.random.gamma(0.5, 0.5, size=K) # variances

    etas = [] # etas (transition matrices un-normalized)
    for j in range(d):
        eta = np.zeros((K, K), dtype=np.float64) 
        for k in range(K):
            for i in range(K):
                shape = gamma + (kappa if i == k else 0.0)
                eta[k, i] = np.random.gamma(shape, 1.0)
        etas.append(eta)
    
    states = np.zeros((d, T_), dtype=np.int64) # states
    for j in range(d):
        states[j, :] = sample_states(y[j, :], x[j, :], etas[j], features[j, :], phis, sigma2s)

    y_pred_est = 0.0
    n_states = []

    likelihood_sums = np.zeros(d*(T_-1), dtype=np.float64)
    log_likelihood_sums = np.zeros(d*(T_-1), dtype=np.float64)
    log_likelihood_2_sums = np.zeros(d*(T_-1), dtype=np.float64)

    # begin sampling
    n_samps_ = 0
    for it in range(n_iter):
        
        # update features
        features, phis, sigma2s, etas = sample_features(y, x, features, phis, sigma2s, etas, alpha, gamma, kappa, params)
        K = features.shape[1]

        # update states 
        for j in range(d):
            states[j, :] = sample_states(y[j, :], x[j, :], etas[j], features[j, :], phis, sigma2s)

        # update auxiliary ys
        for j in range(d):
            y[j, -1] = np.random.normal(phis[states[j, -1]]*x[j, -1], np.sqrt(sigma2s[states[j, -1]]))

        # update phis and sigma2s
        for k in range(K):
            sum_yy, sum_xy, sum_xx, n = get_summary_states_for_state(y, x, states, k)
            phi_, sigma2_ = sample_phi_sigma2_post(sum_yy, sum_xy, sum_xx, n, params)
            phis[k] = phi_ 
            sigma2s[k] = sigma2_

        # update etas 
        for j in range(d):
            transition_counts = get_transition_counts(states[j, :], K)
            
            for k1 in range(K):
                for k2 in range(K):
                    a_post = gamma + transition_counts[k1, k2] + (kappa if k1 == k2 else 0.0)
                    etas[j][k1, k2] = np.random.gamma(a_post, 1.0)
        
        # update alpha 
        alpha = np.random.gamma(alpha_a + features.shape[1], scale=1.0/(alpha_b + np.sum(1.0/np.arange(1, d+1, dtype=np.float64))))

        # update gamma
        for th in range(5):
            gamma = update_gamma(gamma, kappa, etas, features, gamma_a, gamma_b, gamma_prop_sd)

        # update kappa
        for th in range(5):
            kappa = update_kappa(kappa, gamma, etas, features, kappa_a, kappa_b, kappa_prop_sd)
        
        temp_states = states[:, :(T_-1)]
        flattened = flatten(temp_states)
        n_state = len(set(flattened))

        pred_est = predictive_density(y_pred, x[:, -1], temp_states, phis, sigma2s, etas, features)
        pred_est = 1.0
        for j in range(d):
            pred_est *= likelihood(y_pred[j], x[j, -1], phis[states[j, -1]], sigma2s[states[j, -1]])
        # if pred_est > 0.00005:
        #     print(0)
        
        if verbose and (it % thin == 0):
            print('Iter, ', it, ', n states: ', n_state, ', alpha: ', alpha, ', gamma: ', gamma, ', kappa: ', kappa, ', pred_est: ', pred_est)
            # print(states[:, T_-1])
            # print(list(states.flatten()))
            # for j in range(d):
            #     print(states[j, 150:T_])
        
        if (it >= burn_in) and (it % thin == 0):

            if store_clusters:
                cluster_samples.append(flattened)
            n_states.append(n_state)
            # predictive density

            y_pred_est += pred_est
            n_samps_ += 1
            
            for j in range(d):
                for t in range(T_-1):
                    i = j*(T_-1) + t
                    s = int(states[j, t])
                    like = likelihood(y[j, t], x[j, t], phis[s], sigma2s[s])
                    likelihood_sums[i] += like
                    log_likelihood_sums[i] += np.log(like)
                    log_likelihood_2_sums[i] += np.log(like)*np.log(like)

    # lppc and waic
    lppd = 0.0
    penalty = 0.0
    for j in range(d):
        for t in range(T_-1):
            i = j*(T_-1) + t
            lppd += np.log(likelihood_sums[i] / (n_samps_))
            penalty += (log_likelihood_2_sums[i] - log_likelihood_sums[i]*log_likelihood_sums[i]/n_samps_) / (n_samps_ - 1)

    print("lppd = ", lppd) 
    print("waic = ", -2*(lppd - penalty))

    y_pred_est /= n_samps_

    return n_states, flatten(states), y_pred_est

def sim_ar_process(d, T_, cp_prob, phis, variances, states, RNG):

    out = np.zeros((d, T_ + 1))
    true_states = np.zeros((d, T_), dtype=np.int64)
    out[:, 0] = RNG.normal(0, 1, size=d)
    for j in range(d):
        state = None
        for t in range(T_):
            if t == 0:
                state = RNG.choice(states[j])
            else:
                if RNG.uniform(0, 1) < cp_prob:
                    state = RNG.choice(states[j])
            
            phi, v = phis[state], variances[state]
            out[j, t+1] = RNG.normal(phi*out[j, t], np.sqrt(v))
            true_states[j, t] = state

    return out[:, 1:], true_states[:, 1:]

def align_states(Z_true, Z_model, d, T):
    Z_true = np.array(Z_true)
    Z_model = np.array(Z_model)
    
    K_true = Z_true.max() + 1
    K_model = Z_model.max() + 1

    # Step 1: Build confusion matrix
    C = np.zeros((K_true, K_model), dtype=int)
    for k in range(K_true):
        for j in range(K_model):
            C[k, j] = np.sum((Z_true == k) & (Z_model == j))

    # Step 2: Hungarian matching (maximize total overlap)
    row_ind, col_ind = linear_sum_assignment(-C)  # negative because scipy minimizes
    # row_ind = true state indices
    # col_ind = corresponding model state indices

    # Step 3: Create mapping from model state -> true state
    model_to_true = dict(zip(col_ind, row_ind))

    # Any unmatched model states get a new label after K_true
    unmatched = set(range(K_model)) - set(col_ind)
    next_label = K_true
    for u in unmatched:
        model_to_true[u] = next_label
        next_label += 1

    # Step 4: Relabel model states
    Z_model_aligned = np.array([model_to_true[j] for j in Z_model])

    # Step 5: Collapse to series × state presence matrices
    A_true = np.zeros((d, K_true), dtype=int)
    B_model = np.zeros((d, next_label), dtype=int)  # include unmatched states

    for n in range(d):
        idx = slice(n*T, (n+1)*T)  # indices for this series
        # True states
        unique_true = np.unique(Z_true[idx])
        A_true[n, unique_true] = 1
        # Model states
        unique_model = np.unique(Z_model_aligned[idx])
        B_model[n, unique_model] = 1

    return A_true, B_model, Z_model_aligned, C, row_ind, col_ind

def pad_matrix(mat, max_cols):
    if mat.shape[1] < max_cols:
        pad_width = max_cols - mat.shape[1]
        mat = np.hstack([mat, np.zeros((mat.shape[0], pad_width), dtype=int)])
    return mat

def produce_matrices(Z_true, Z_models, d, T_):

    n = len(Z_models)

    A_true, B_model_sum = None, None

    for i in range(n):
        A_true_, B_model_, _, _, _, _ = align_states(Z_true.flatten(), np.array(Z_models[i], dtype=np.int64), d, T_)
        if i == 0:
            max_col = max(A_true_.shape[1], B_model_.shape[1])
            A_true = pad_matrix(A_true_, max_col)
            B_model_sum = pad_matrix(B_model_, max_col).astype(np.float64)
        else:
            max_col = max([A_true_.shape[1], B_model_.shape[1], A_true.shape[1], B_model_sum.shape[1]])
            A_true = pad_matrix(A_true, max_col)
            B_model_sum = pad_matrix(B_model_sum, max_col)
            B_model_ = pad_matrix(B_model_, max_col)
            
            B_model_sum += B_model_.astype(np.float64)
    
    return A_true, B_model_sum / n

BLOCK_SIZE = 2
T__ = 800
def get_scenario_slice(task_id):
    tasks_per_scenario = (50*T__)/BLOCK_SIZE
    scenario_num = task_id // tasks_per_scenario 
    tasks_per_dataset = (T__) / BLOCK_SIZE
    dataset_num = (task_id % tasks_per_scenario) // tasks_per_dataset 
    slice_num = (task_id % tasks_per_dataset)
    return int(scenario_num), int(dataset_num), int(slice_num)

def plot_data(y: np.array, map_clusters: np.array, max_cluster=10) -> None:
    d, T_ = map_clusters.shape
    fig, axs = plt.subplots(d, 1, sharex=True)
    for j in range(d):
        x_, y_ = np.arange(1, T_+1), y[j, (y.shape[1] - T_):]
        for i in range(len(y_) - 1):
            color = plt.cm.tab10(map_clusters[j, i] / max_cluster)  # Normalize the value for colormap
            axs[j].plot(x_[i:i+2], y_[i:i+2], lw=1.8, color=color)
        # axs[j].vlines([327], np.min(y_)-0.1, np.max(y_)+0.1, color='blue', linestyle='dashed')
        axs[j].set_ylabel(j+1)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__1":
    task_id = int(sys.argv[1])
    
    scen_num, data_num, slice_num = get_scenario_slice(task_id)
    start_ind, end_ind = slice_num*BLOCK_SIZE, (slice_num+1)*BLOCK_SIZE
    print((scen_num, data_num, slice_num, start_ind, end_ind))

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from time import perf_counter

    RNG = np.random.default_rng(seed=1)
    states = {
        0 : [0,   2,    4],
        1 : [  1,   3    ],
        2 : [0,   2,    4],
        3 : [  1,   3    ],
        4 : [0,   2,    4]
    }
    
    RNG = np.random.default_rng(seed=2)
    states = {
        0 : [0, 1, 2       ],
        1 : [   1, 2, 3    ],
        2 : [       2, 3, 4],
        3 : [   1, 2, 3    ],
        4 : [0, 1, 2       ]
    }
    
    RNG = np.random.default_rng(seed=2)
    states = {
        0 : [0, 1,    3],
        1 : [0, 1, 2],
        2 : [0,    2, 3],
        3 : [0, 1,       4],
        4 : [   1,    3, 4]
    }

    d = len(states)
    y, true_states = sim_ar_process(d, 800, 0.04, np.array([0.8, 0, -0.8, 0.5, -0.5]), np.array([0.5, 6, 0.5, 2, 2]), states, RNG)
    print(y.shape)
    
    # y = np.array([[-0.34055284294387306, -0.0833102724674801, 0.21557980687744446, 0.34108807451746304, 0.5803357603719402, 0.4408472103382198, 0.6961855956217116, 0.48033866057699615, 0.3037086742098612, 0.13553635413918988, 0.625344876378168, -0.3816962342434851, -0.5546218166073134, -0.39049866943530026, -0.9074591838379109, 0.20467253829204535, 0.5157369706509717, -0.35987431184195184, -0.2693798367469829, -0.5840551839625657, -0.9980254770966979, -0.850481415652301, -1.0185791032442282, -0.47113771577864116, -0.8331390578959306, -1.0005451613398637, -1.4363843536445995, -1.3027776455878797, -1.3294496170125156, 0.7054105928407752, -0.5350772671331573, 0.5993223462849542, -1.363783794831933, 1.8397288991166059, -2.06116330082373, 1.9817702288454278, -1.9562010609865248, 1.7055755725322468, -1.147356104255235, 0.21719587761216086, 0.24641822844268285, -0.541898088552659, 0.7104950565854862, -0.20132445286976253, 0.1337778380985372, 0.4074713459779421, -0.43605187392581657, -0.3202277939018589, -0.03743549171002397, -0.029104099478142027, 0.6356154921159395, -1.0099145146157682, 1.7856151219123777, -1.9513360835383007, 1.660496809943146, -1.4774830377302812, 0.9724448734407711, -1.3921337472741853, 2.0148891290418325, -3.046514831813308, 4.015812256373541, -4.239578910137181, 3.395260224094309, -3.229792935026767, 2.704162460827531, -2.5220128441915657, 2.1098984309007984, -1.8956510948758443, 1.1596388007406384, -1.0702670308192452, 1.2190085376961302, -1.2113753676655963, 1.2314457520212374, -0.6916298784842829, 0.09442741547876177, 0.983201611050388, -1.0801541221230107, 0.38437075745031934, -1.414845860750576, 1.723144483413522, -1.865507497013841, 2.029032622887782, -0.8398924336341083, 0.4596981403942311, -0.449977599164714, 0.3899659326335252, -1.186394369677729, 0.7803852982494401, -0.3834711886037783, 0.5916381444931218, -0.9985844827297856, 0.9212527216395263, -1.3987747846470615, 1.3023869304683076, -1.0878330339162197, 1.5921850306827063, -2.7886092463318217, 1.7010145719367957, -1.4468601801792038, 0.769611797759634, 0.317386065591552, -0.09923989809923872, 0.0775309398589488, 0.8637949114172554, -0.9254573277742772, 0.5091312557887853, -0.740038053423389, 0.08076293053089878, -0.17613309983393105, 0.18686476729743542, 0.20842741365199854, -0.25590947043474077, -0.032938896666351775, 0.2890374674386987, -0.6330619197889913, 0.7710424107823584, -1.7035710749485236, 1.662995696899701, -1.111535085812483, 0.4372875191545845, -0.4827939829387336, -0.368359444517359, 0.030194192793759123, 0.28224632076919154, -0.09159244580582049, 0.052563020985278354, -0.4206331608781997, 0.14361979525060803, -0.09074160481429583, -0.03287222647645355, 0.33947778698312464, -0.8209620763025787, 0.05833616951978715, -0.07993411157704035, -0.38545447548730943, 0.5134998702890785, 1.0878712661971215, -1.3440016659401883, 1.1896855786283274, -0.7572533307191884, 0.4565548609092989, -0.16790707618867304, 0.37031065067325997, -0.670745893493893, 0.49086838876396066, 0.058923562952517494, 0.33801105462268255, -0.172759390617899, 0.48950716484218143, -0.5954002261167342, 0.4569418524098846, -0.7619873164956663, 0.5403298183210545, -0.486196015011387, 0.913587347510149, -0.10386034271466493, -0.310689542499864, 0.2222618922663092, -0.2175830951041017, 0.8918939858664409, -1.1236834496277839, 0.8190998675609071, -1.2594453636104967, 1.039913324718679, -0.4663154404076505, 0.4276944981668303, -1.0525192351627073, 0.9319755768898048, -0.03859995340399469, 0.1652140475903095, 0.14740570287096547, -0.38036491608586565, 0.6490831176949652, -0.9805902754193744, 1.3247826935021265, -2.0342116225724425, 2.0389147001719357, -2.003336963343083, 1.2724708070079374, -1.3904138811211721, 1.75419455121725, -1.604389835844897, 2.0387832611807206, -2.444543707067878, 2.5726339410506593, -3.154945095322689, 3.5061361798340873, -3.755252853813042, 3.88893189887113, -3.229975122837986, 3.6663521275635547, -2.804593258823166, 2.4316945082098385, -1.6189710836462425, 1.0811974096348338, -0.5877620304575433, 0.40015466989588455, -1.2076354059007377, 1.204160829924304, -0.71252162199434, 0.3653630490984275, 0.6649281718058755, -0.317120227022157, 0.47427098633987474, -0.9282127312690984, 1.2188004026273205, -1.2601203006375559, 1.6602791990485657, -1.8542831100972064, 1.729493428747344, -1.5278357548514268, 1.6825548112717683, -1.6890329033615026, 1.2143918056434393, -0.8818749394529378, 0.54096676383316, -0.3987876116386317, 0.0226724750054631, 0.0040623947162739815, -0.5675997433739648, 0.1728367671009844, 0.5322160585942033, -0.3000600077192513, 0.297295612407449, 0.7979200656949521, -0.858334131272819, -1.1637000061202454, -0.9495226594056629, -0.8285541584884, 0.1380457979228087, 0.9281406619386471, 0.6265589482649745, 0.5753419624609424, 0.4184171497173813, 0.5936660049867516, 0.5296055734899795, 0.6070427657810855, 0.6283159083828804, 0.8604558693244253, 0.6548858527296884, 1.0903401348212909, 2.257735832516759, 2.5317924639425065, 2.211728294377367, 1.3992540011760344, 1.8144745783278455, 1.719455158179938, 1.8280245087587297, 1.4231730703186452, 1.294918918063261, 2.2532259883288654, 1.4891431410721485, 1.8954127278442774, 2.040199661541956, 1.9660149009118308, 1.4633602218267376, 0.6716469999527547, 0.5233590385672898, 0.5968427300129954, 0.0035113405256984587, 0.2876746147625397, 1.028865257239921, 0.5416900529425761, 1.1862323392020908, 0.07267894574700662, -0.21812234200572755, 0.10736347911614316, -0.5264115011300029, 0.46327031982359956, 0.9124970116830109, 1.3130630078369854, 1.557264025554813, 1.9669904719390567, 1.225351531888975, 1.7119271136059706, 1.2611644631530405, 0.38722504661485624, 1.004001193446384, 1.0248862318203837, 1.0420877589006732, 1.3848017832364419, 1.712593371914751, 1.628464591772321, 1.3827277231054116, 1.7726769877088684, 1.6852573898992622, 1.7159844763248726, 1.8869738868036687, 1.7163071315477536, 0.6395475413665103, 0.5121767687099958, 0.8056661679087571, 1.2035641803234256, 0.7309630046175636, 0.20064421260372073, -0.34939858813060065, -1.8545476088423696, -1.8351380273396376, -2.3928759569045774, -2.4271384460315257, -1.9969930788619736, -2.2246195058941476, -1.054169170825781, -0.7933959814134603, -0.7253921052816223, -1.359477285899294, -0.13417318601028816, -0.11516001400263459, -0.03940613857649222, -0.15729018462543698, 0.21886074659152585, 1.0550986716403497, 0.7971090214091565, 0.2947449455160664, 0.10340954383937556, 1.0142923922645164, 0.74829752595657, -0.26821211729158023, 0.23398756993526382, -0.027826173605005416, 0.30536069200877697, 0.4758683279519663, 0.4432506092453361, 0.03444611276798493, -0.8574344009150359, -1.0469488255973194, -0.8551705643515992, -1.3045813044242707, -1.3759563666800625, -1.9546328903834245, -1.2506015346555435, -1.245845717412177, -1.0297373403820085, -1.9696074403901407, -0.8639440195749314, -0.19665387572626447, -1.255783902234686, -1.3393451654173238, -0.8129923119140325, -0.994397901081717, -1.3309940948700616, -1.6681249200047055, -2.0516864002788955, -1.7193640102430536, -1.958650597978673, -1.3429531389234053, -1.7420532523546912, -1.3352932651720348, -1.3566443821971934, -1.5864020790858855, -2.2751593580356073, -2.322114133444297, -3.8643052055988356, -3.7441909296356797, -3.6591681491956347, -4.44748800085189, -4.553294840500165, -3.317622397953375, -3.272164860689377, -3.5214693616401176, -3.1678463515513116, -2.37584329207429, -2.3324168801251877, -2.4950099824271197, -2.554537201844096, -1.8089513224923572, -1.2085180588475, -0.45151037944305505, -0.26916854345511065, -0.8177323130344742, -0.6017725260727591, -0.2489290401421798, -0.3135353781902288, -0.5468199484826031, -0.015890907903395035, 0.5774083162925318, -0.12935934243465919, -0.7121823108779659, -0.04161111526604233, -0.34371046066926286, -0.2141144132690141, -0.12540190031233506, -0.3929831481071954, -0.276762511753992, -0.6693176575769055, -0.5007005320501074, -0.9546194783093436, -0.3997680883944698, -0.7408829425891355, -0.7028899439291021, -0.8170832728655677, -0.8693478273631923, -0.916832845345517, -0.9754971736938608, -1.2164858287037057, -1.8623764328493508, -1.7246286548549308, -2.1620750444703427, -1.9277741459959838, -0.8465294057661742, -1.3520184211143507, -0.5724549946406595, -0.1014400066356889, -1.776633025684182, 0.3440325538519612, -1.0536869104243363, 1.7259053255339913, -0.03816312861770822, 0.3076118347894994, -0.7447583246892102, 0.33613691249304456, -0.3612801099148547, 0.006993433143757166, 0.5026893610561698, -0.6104025352016397, 0.6586253741356151, -0.5096650941383045, 0.6628004092197721, -0.3315621460936538, 0.4160360803144554, -0.5111022849641463, -0.15852320489205546, 0.08684005508435594, 0.21858387974450247, 0.37915869537000596, -0.33883737435075645, 0.06306514609157635, 0.3823896949178791, -0.5500395502509369, 0.526872410641821, 0.30346482576594214, 1.602699140443831, -1.6418734430466042, 1.8412822912708622, -1.9969923788992054, 1.8102050729566914, -0.9459036349039585, 0.34915670794814047, -0.20070225685765514, 0.30441965093888446, -0.07040400288729323, -0.6134337547243061, 0.667593256503505, 0.275298061419481, 0.16892921638554317, 0.9607670652389377, -1.1611783301353746, 0.7928961875429864, -0.6180130809503164, -0.5270943068038384, -0.06136310241914078, 0.3550056411592313, 0.09191239820965225, -0.983918618634125, 0.09211777231513318, 0.3666929941694931, 0.5645562106169109, -1.1068844895750414, 1.384583404479989, -1.2532580446061372, 1.7217053259077741, -1.7708772204736953, 0.9404838686274111, 0.06732380173938257, 0.6081890438632627, -0.9071826203193029, 0.844931379070569, -1.182789106043632, 0.31227622705742075, -0.0950081818992583, -0.828285759156134, 0.7259041468612677, -0.615396175951088, 0.8040669239472319, -0.26199301399923547, 0.2070686881747716, -0.5769776055688677, 0.27384343432475444, 0.14194996159943232, 0.801838585744568, -0.26612556087319844, -0.16072015226967984, -1.3691596739213179, 1.6518881973872437, -1.4789074106205704, 1.2037776397504063, -1.6573845663270408, 0.8595243390939334, -0.3371963208534123, 0.6281993264384607, 0.5647568730090148, -0.8887752763780478, 1.2406271649646312, -0.569571482132343, 0.39180185502921083, -0.8568466521788551, 1.2470736604648551, -1.299531515428496, 0.9900322960040326, -2.100278750704386]])

    # y = np.loadtxt("OneDrive - King's College London/Documents/Code/SNCoRM/eeg_dat.csv", delimiter=',')

    y = np.loadtxt("OneDrive - King's College London/Documents/Code/SNCoRM/seis_dat.csv", delimiter=',')
    y_pred = y[:, 178]
    y = y[:, :178]

    params = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    # params = np.array([1.6, 2.64, 1.312])
    print('compile done')

    t1 = perf_counter()
    error = True
    while error:
        try: 
            n_states, cluster_samples, y_pred = sample(y, 1000, 0, y_pred, params, verbose=True, store_clusters=True)
            error=False 
        except Exception as ex:
            print(ex)
            print('retrying')
            continue
    
    t2 = perf_counter()
    print(f'Took {round(t2-t1, 2)}s')
    print(f'Predictive estimate: {y_pred}')

    plot_data(y, np.array(cluster_samples).reshape(y.shape[0], y.shape[1]-1))

    plt.plot(n_states)
    plt.show()

    A_true, B_model = produce_matrices(true_states, cluster_samples, d, true_states.shape[1])

    print(B_model)
    
    A_true_padded = A_true
    B_model_padded = B_model
    max_states = A_true.shape[1]

    # True states
    # Colormap: 0 = gray (absent), 1 = white (present)
    # cmap = mcolors.ListedColormap(['black', 'white'])
    cmap = plt.cm.gray
    # plt.figure(figsize=(12,5))
    fig, ax = plt.subplots(1, 2, width_ratios=(0.5, 0.62))
    # plt.subplot(1,2,1)
    ax[0].imshow(A_true_padded, aspect='auto', cmap=cmap, vmin=0, vmax=1)
    ax[0].set_title("True states per series")
    ax[0].set_xlabel("State")
    ax[0].set_ylabel("Series")
    ax[0].set_xticks(ticks=np.arange(max_states), labels=np.arange(1, max_states+1))
    ax[0].set_yticks(ticks=np.arange(d), labels=np.arange(1, d+1))

    # plt.subplot(1,2,2)
    im = ax[1].imshow(B_model_padded, aspect='auto', cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax[1], label='Probability')
    ax[1].set_title("Estimated model states per series")
    ax[1].set_xlabel("State")
    ax[1].set_ylabel("Series")
    ax[1].set_xticks(ticks=np.arange(max_states), labels=np.arange(1, max_states+1))
    ax[1].set_yticks(ticks=np.arange(d), labels=np.arange(1, d+1))
    plt.tight_layout()
    plt.show()