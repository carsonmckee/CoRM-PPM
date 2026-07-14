#include "sncorm.h"
#include <omp.h>
#include <iostream>
#include <vector>

void write_eeg_results(const Samples& samps){
    write_csv("eeg_results/sncorm_states.csv", samps.states);
    write_csv("eeg_results/sncorm_runs.csv", samps.runs);
    write_csv("eeg_results/sncorm_phis.csv", {samps.phi});
    write_csv("eeg_results/sncorm_Ms.csv", {samps.M});
    write_csv("eeg_results/sncorm_n_states.csv", {samps.n_states});
    // write_csv("eeg_results/sncorm_unique_per_series.csv", samps.unique_per_series);
    std::cout << "wrote eeg results" << std::endl;
}

// g++ -std=c++17 -Ofast -fopenmp -funroll-loops sncorm2.cpp sncorm_eeg.cpp -o sncorm_eeg.exe
int main(){
    
    // omp_set_num_threads(8);

    int t = 1250;
    std::vector<std::vector<double>> y = read_csv("eeg_dat.csv");
    
    std::vector<double> y_preds(y.size());

    for(int j=0; j<y.size(); j++){
        y_preds[j] = y[j][t];
        y[j].resize(t);
    }

    print_vec(y_preds);
    std::cout << y.size() << std::endl;
    std::cout << y[0].size() << std::endl;
    std::cout << "start \n";
    
    Samples samps = sample(
            y, // data
            2000, // n_samples
            1000, // burn_in
            {1.0, 1.0, 1.0}, // params
            y_preds, // posterior predictive values
            2.0, // M_a 
            2.0, // M_b
            2.0, // phi_a
            2.0, // phi_b
            0.1, // u_upper
            0.1, // M_prop_sd
            0.3, // phi_prop_sd
            true, // posterior_checks
            20, // h
            true // verbose
        );
        
    // write_csv("eeg_results/sncorm_states_2.csv", samps.states);
    
    // write_csv("eeg_results/sncorm_states.csv", samps.states);
    /*
    write_csv("eeg_results/sncorm_residuals_0.csv", samps.residuals[0]);
    write_csv("eeg_results/sncorm_residuals_1.csv", samps.residuals[1]);
    write_csv("eeg_results/sncorm_residuals_2.csv", samps.residuals[2]);
    write_csv("eeg_results/sncorm_residuals_3.csv", samps.residuals[3]);
    write_csv("eeg_results/sncorm_residuals_4.csv", samps.residuals[4]);
    write_csv("eeg_results/sncorm_residuals_5.csv", samps.residuals[5]);
    */

    std::string acf1_path = "eeg_results/sncorm_acf1s_" + std::to_string(t) + ".csv"; 
    std::string acf2_path = "eeg_results/sncorm_acf2s_" + std::to_string(t) + ".csv"; 
    std::string variance_path = "eeg_results/sncorm_variances_" + std::to_string(t) + ".csv"; 
    
    write_csv(acf1_path, samps.acf1s);
    write_csv(acf2_path, samps.acf2s);
    write_csv(variance_path, samps.variances);
}
