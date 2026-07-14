import sys
import numpy as np

from fox import sample

points_per_task = 10
T_ = 600
n_datasets = 50
n_scenarios = 4

if __name__ == "__main__":

    task_id = int(sys.argv[1])

    scenario_ = int(task_id / ((T_/points_per_task) * 50))
    dataset = int((task_id - scenario_ * (T_/points_per_task)*50) / (T_/points_per_task))
    block = int(task_id % (T_ / points_per_task))
    t = int(block * points_per_task + 2)
    scenario = str(scenario_)
    
    path = "SimData/" + scenario + "_" + str(dataset) + ".csv"
    print(path)
    y_ = np.loadtxt(path, delimiter=',')
    d = y_.shape[0]
    print(y_.shape)
    log_y_preds = np.zeros(points_per_task)

    for i, t_ in enumerate(range(t, t+points_per_task)):
        print(f't_ = {t_}')
        y = y_[:, :t_]
        y_pred = y_[:, t_]
        print(y_pred)
        print(y.shape)
        error = True
        while error:
            try:
                _, _, y_pred_est = sample(y, 8000, 1000, y_pred, np.array([1.0, 1.0, 1.0], dtype=np.float64), verbose=False, thin=1)
                log_y_preds[i] = np.log(y_pred_est)
                print(f'y_pred = {y_pred_est}')
                error = False
            except Exception as ex:
                print(ex)
                print('retrying') 

    file_name = f"SimResults/fox_{scenario}_{dataset}_{block}.csv"
    np.savetxt(file_name, log_y_preds.reshape(1, -1), delimiter=',')
    print(f'wrote to {file_name}')

