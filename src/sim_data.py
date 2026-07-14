import numpy as np 

S1_F = np.array([
    [1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0], 
    [1, 0, 1, 0, 1], 
    [0, 1, 0, 1, 0], 
    ])

S2_F = np.array([
    [1, 1, 0, 1, 0],
    [1, 1, 1, 0, 0], 
    [1, 0, 1, 1, 0], 
    [1, 1, 0, 0, 1],
    ])

PHIS = np.array([0.8, 0, -0.8, 0.5, -0.5])
SIGMA2S = np.array([0.5, 6, 0.5, 2, 2])

def sim_trans_matrices(states, RNG):
    gamma, kappa = 1, 100
    etas = []
    alpha = gamma*np.ones(5)
    for j in range(4):
        eta = np.zeros((5, 5))
        for k in range(5):
            alpha[k] += kappa 
            eta[k, :] = RNG.gamma(alpha, np.ones(5), size=5) * states[j, :]
            alpha[k] -= kappa 
            eta[k, :] = eta[k, :] / np.sum(eta[k, :])
        etas.append(eta)
    return etas

def S_HMM(T_, states, RNG):
    pis = sim_trans_matrices(states, RNG)
    
    out = np.zeros((4, T_+1))
    out[:, 0] = RNG.normal(0, 1, size=4)
    states_out = np.zeros((4, T_+1), dtype=np.int64)
    for j in range(4):
        active = np.where(states[j, :] == 1)[0]
        state = RNG.choice(active)
        for t in range(1, T_+1):
             state = RNG.choice(5, p=pis[j][state, :])
             states_out[j, t] = state
             out[j, t] = RNG.normal(out[j, t-1]*PHIS[state], np.sqrt(SIGMA2S[state]))
    
    return out[:, 1:], states_out[:, 1:]

def S_Mix(T_, states, RNG):
    qs = (1/50)*np.ones(4)
    
    out = np.zeros((4, T_+1))
    out[:, 0] = RNG.normal(0, 1, size=4)
    states_out = np.zeros((4, T_+1), dtype=np.int64)
    for j in range(4):
        active = np.where(states[j, :] == 1)[0]
        state = RNG.choice(active)
        for t in range(1, T_+1):
             if RNG.uniform(0, 1) < qs[j]:
                state = RNG.choice(5, p=(states[j, :] / np.sum(states[j, :])))
             states_out[j, t] = state
             out[j, t] = RNG.normal(out[j, t-1]*PHIS[state], np.sqrt(SIGMA2S[state]))

    return out[:, 1:], states_out[:, 1:]


if __name__ == "__main__":
    
    RNG = np.random.default_rng(seed=1)

    scenarios = [S1_F, S2_F]
    BASE_DATA_PATH = "c:/Users/k2259011/OneDrive - King's College London/Documents/Code/SNCoRM/SimData"
    BASE_STATES_PATH = "c:/Users/k2259011/OneDrive - King's College London/Documents/Code/SNCoRM/SimStates"
    for s in range(4):

        for i in range(50):

            if s in {0, 1}:
                # HMM
                data, states = S_HMM(610, scenarios[s], RNG)
                np.savetxt(f'{BASE_DATA_PATH}/{s}_{i}.csv', data, delimiter=',')
                np.savetxt(f'{BASE_STATES_PATH}/{s}_{i}.csv', states, delimiter=',')
                
            else:
                # Mixture
                data, states = S_Mix(610, scenarios[s-2], RNG)
                np.savetxt(f'{BASE_DATA_PATH}/{s}_{i}.csv', data, delimiter=',')
                np.savetxt(f'{BASE_STATES_PATH}/{s}_{i}.csv', states, delimiter=',')
