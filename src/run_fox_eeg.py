import sys
import numpy as np

from fox import sample

if __name__ == "__main__":
    
    task_id = int(sys.argv[1])
    # task_id = 177
    t = task_id + 2

    y = np.loadtxt('eeg_dat.csv', delimiter=',')
    # y = y[:3, :]
    # y = np.loadtxt('eeg_dat_first.csv', delimiter=',')
    # y = np.loadtxt('eeg_dat_second.csv', delimiter=',')

    y_pred = y[:, t]

    print(y_pred)
    y = y[:, :t]
    print(y[:, y.shape[1]-1])
    
    print(f'task id: {task_id}, t: {t}')
    print(y.shape)
    
    print('start')
    
    params = np.array([1, 1, 1]) 
    error = True
    while error:
        try:
            n_states, cluster_samples, y_pred = sample(y, 50000, 2000, y_pred, params, verbose=True, thin=1)
            error=False
        except Exception as ex:
            print(ex)
            print('retrying')
    np.savetxt(f'eeg_results/y_preds/fox_{task_id}.csv', np.array([y_pred]))
    print(f'y_pred = {y_pred}')
    
    if t == 1278:
        np.savetxt(f'eeg_results/fox_n_states.csv', n_states)
        np.savetxt(f'eeg_results/fox_states.csv', cluster_samples)
        print('wrote model data')