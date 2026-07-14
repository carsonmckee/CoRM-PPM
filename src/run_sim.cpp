#include "sncorm.h"
#include <iostream>
#include <vector>
#include <string>

int points_per_task = 10;
int T = 600;
int n_datasets = 50;
int n_scenarios = 4;

// g++ -std=c++17 -Ofast -fopenmp -funroll-loops run_sim.cpp sncorm2.cpp -o run_sim.exe
int main(int argc, char* argv[]){
    int task_id = std::stoi(argv[1]);

    int scenario_ = static_cast<int>(task_id / ((T/points_per_task) * 50));
    int dataset = static_cast<int>((task_id - scenario_ * (T/points_per_task)*50) / (T/points_per_task));
    int block = task_id % (T / points_per_task);
    int t = block * points_per_task + 2;
    std::string scenario = std::to_string(scenario_);

    std::cout << "scenario = " << scenario << std::endl;
    std::cout << "dataset = " << dataset << std::endl;
    std::cout << "block = " << block << std::endl;
    std::cout << "t = " << t << std::endl;

    std::string path = "SimData/" + scenario + "_" + std::to_string(dataset) + ".csv";
    std::cout << path << std::endl;
    std::vector<std::vector<double>> y_ = read_csv(path);
    std::vector<std::vector<double>> y;
    int d = y_.size();

    std::vector<double> y_pred(d);

    std::cout << y_.size() <<  ", " << y_[0].size() << std::endl;

    std::vector<double> log_y_preds(points_per_task, 0);

    int counter = 0;
    for(int t_=t; t_<t+points_per_task; t_++){
        std::cout << "t_ = " << t_ << std::endl;
        y = y_;
        for(int j=0; j<d; j++){
            y_pred[j] = y_[j][t_];
            y[j].resize(t_);
        }
        print_vec(y_pred);
        std::cout << "(" << y.size() << ", " << y[0].size() << ")\n";
        Samples samps = sample(
            y, 
            1200, 
            200, 
            {1.0, 1.0, 1.0},
            y_pred, // posterior predictive values
            2.0, // M_a 
            2.0, // M_b
            2.0, // phi_a
            2.0, // phi_b
            0.1, // u_upper
            0.1, // M_prop_sd
            0.3, // phi_prop_sd
            false, // posterior_checks
            1, // h
            false, // verbose
            5e-4
        );
        
        log_y_preds[counter] = std::log(samps.y_pred);
        std::cout << "y_pred = " << samps.y_pred << std::endl;
        counter += 1;
    }
    std::string file_name = "SimResults/sncorm_" + scenario + "_" + std::to_string(dataset) + "_" + std::to_string(block) + ".csv";
    write_csv(file_name, {log_y_preds});
    std::cout << "wrote to " << file_name << std::endl;
}