#include "sncorm.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>

void write_eeg_results(const Samples& samps){
    write_csv("eeg_results/sncorm_states.csv", samps.states);
    write_csv("eeg_results/sncorm_runs.csv", samps.runs);
    write_csv("eeg_results/sncorm_phis.csv", {samps.phi});
    write_csv("eeg_results/sncorm_Ms.csv", {samps.M});
    write_csv("eeg_results/sncorm_n_states.csv", {samps.n_states});
    // write_csv("eeg_results/sncorm_unique_per_series.csv", samps.unique_per_series);
    std::cout << "wrote eeg results" << std::endl;
}

// to compile:
// g++ -std=c++17 -Ofast -funroll-loops run_eeg.cpp sncorm.cpp -o run_eeg.exe
int main(int argc, char* argv[]){

    int task_id = std::stoi(argv[1]);
    int t = task_id + 2;
    
    std::vector<std::vector<double>> y = read_csv("eeg_dat.csv");
    
    int n_samples = 20000;
    int burn_in = 2000;

    std::cout << "n_samples: " << n_samples << ", burn_in: " << burn_in << std::endl;

    std::vector<double> y_pred(y.size());
    for(int j=0; j<y.size(); j++){
        y_pred[j] = y[j][t];
        y[j].resize(t);
    }

    std::cout << "task id: " << task_id << ", t: " << t << std::endl;
    std::cout << y.size() << std::endl;
    std::cout << y[0].size() << std::endl;
    print_vec(y_pred);
    std::cout << "start \n";
    
    Samples samps = sample(
            y, 
            n_samples, // iterations
            burn_in, // burn in
            {1.0, 1.0, 1.0},  // params
            y_pred, // posterior predictive values
            2.0, // M_a 
            2.0, // M_b
            2.0, // phi_a
            2.0, // phi_b
            0.2 // u_upper
        );
    
    std::string path = "eeg_results/y_preds/sncorm_" + std::to_string(task_id) + ".csv";
    std::ofstream file(path);
    if (!file) return 1;
    
    file << samps.y_pred << "\n";  // write the value as a row
    
    std::cout << "y_pred = " << samps.y_pred << std::endl;
    std::cout << "wrote to path: " << path << std::endl; 
    
    if (t == 1278){

        write_eeg_results(samps);

        std::cout << "wrote model data" << std::endl;
    }
    return 0;
}
