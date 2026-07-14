#include "sticky_ncorm_cpp.h"
#include <omp.h>
#include <iostream>
#include <vector>
#include <string>

/*
void write_eeg_results(const Samples& samps, const std::string& id){
    write_csv("eeg_results/SNCORM/states_" + id + ".csv", samps.states);
    write_csv("eeg_results/SNCORM/runs_" + id + ".csv", samps.runs);
    write_csv("eeg_results/SNCORM/phis_" + id + ".csv", {samps.phi});
    write_csv("eeg_results/SNCORM/Ms_" + id + ".csv", {samps.M});
    write_csv("eeg_results/SNCORM/n_states_" + id + ".csv", {samps.n_states});
    write_csv("eeg_results/SNCORM/unique_per_series_" + id + ".csv", samps.unique_per_series);
    std::cout << "wrote eeg results with id " + id << std::endl;
}
*/

void write_eeg_results(const Samples& samps){
    write_csv("eeg_results/SNCORM/states.csv", samps.states);
    write_csv("eeg_results/SNCORM/runs.csv", samps.runs);
    write_csv("eeg_results/SNCORM/phis.csv", {samps.phi});
    write_csv("eeg_results/SNCORM/Ms.csv", {samps.M});
    write_csv("eeg_results/SNCORM/n_states.csv", {samps.n_states});
    write_csv("eeg_results/SNCORM/unique_per_series.csv", samps.unique_per_series);
    std::cout << "wrote eeg results" << std::endl;
}

// g++ -fopenmp -Ofast eeg_analysis.cpp sticky_ncorm_cpp.cpp -o eeg.exe 
int main(){
    
    // std::string id = argv[1];
    // std::cout << "id = " << id << std::endl;
    std::vector<std::vector<double>> y = read_csv("eeg_dat.csv");
    
    std::cout << y.size() << std::endl;
    std::cout << y[0].size() << std::endl;
    std::cout << "start \n";

    Samples samps = sample(
            y, 
            50, 
            0, 
            1000, 
            {1.0, 1.0, 1.0}
        );
    
    /*
    in real data:
    1. for each series plot: num unique states, number of state changes
    2. create pairs grid showing number of shared states between twi series
    */
    
    write_eeg_results(samps);
    std::cout << samps.y_pred << std::endl;
}
