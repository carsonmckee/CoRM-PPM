#ifndef SNCORM_H
#define SNCORM_H

#include <vector>
#include <random>
#include <cstddef>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <chrono>

static double PI = 4*std::atan(1.0);
static double log2pi = std::log(2.0*PI);

struct VectorHash {
    std::size_t operator()(const std::vector<int>& v) const {
        std::size_t seed = v.size();
        for(auto& i : v) {
            seed ^= std::hash<int>{}(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

double sample_beta(const double& alpha, const double& beta);

void write_csv(const std::string& filename,
               const std::vector<std::vector<double>>& data);

void write_csv(const std::string& filename,
               const std::vector<std::vector<int>>& data);

int countUnique(const std::vector<int>& v);

std::vector<int> sample_ancestors(const int& d, 
                                  const int& T,
                                  const std::vector<double>& weights, 
                                  const std::vector<std::vector<int>>& ancestors, 
                                  const std::vector<std::vector<int>>& x_parts, 
                                  const std::vector<std::vector<int>>& s_parts);

std::vector<std::vector<double>> read_csv(const std::string& filename);

struct Samples {
    int n_particles;
    int n_samples;
    int burn_in;
    double y_pred;
    std::vector<double> M;
    std::vector<double> phi;
    std::vector<std::vector<int>> runs;
    std::vector<std::vector<int>> states;
    std::vector<int> n_states;
    std::vector<std::vector<double>> shared_states;
    std::vector<std::vector<int>> unique_per_series;
    std::vector<std::vector<std::vector<double>>> residuals;
    std::vector<std::vector<double>> acf1s;
    std::vector<std::vector<double>> acf2s;
    std::vector<std::vector<double>> variances;
};

void print_vec(const std::vector<double>& arr);

void print_vec(const std::vector<int>& arr);

int sum(const std::vector<int>& arr);

double sum(const std::vector<double>& arr);

double dnorm(const double& x, const double& mu, const double& sd, const bool& log=false);

double dgamma(const double& x, const double& a, const double& b, const bool& log=false);

double rgamma(double a, double b);

std::pair<double, double> sample_posterior(
    double sum_yy,
    double sum_xy,
    double sum_xx,
    double n,
    double lam,
    double a,
    double b);

double log_g(const double& sum_yy,
             const double& sum_xy,
             const double& sum_xx, 
             const double& n,
             const std::vector<double>& params);

double directing_intensity(const double& z, const double& phi, const double& M);

double log_tau(const double& a, const double& z, const double& v, const double& phi);

double kappa(const std::vector<int>& alpha, 
             const std::vector<double>& v_sums, 
             const double& phi, 
             const double& M,
             double dx=2e-4);

std::vector<int> sample_multinomial(const std::vector<double>& W, size_t N);

int multinomial_sample(const std::vector<double>& weights, const int& n_parts);

double M_f_z_t(const double& z, const double& t, const double& phi);

double psi(const std::vector<double>& v, const double& phi, const double& M, double dx=1e-5);

std::vector<int> flatten(const std::vector<std::vector<int>>& arr, const int& T);

std::vector<int> order_of_appearance(const std::vector<int>& arr);

double update_M(double& curr_M, 
                const std::vector<double>& v, 
                const double& phi, 
                const std::unordered_map<int, std::vector<int>>& n, 
                const double& a, 
                const double& b, 
                const double& prop_sd);

double update_phi(double& curr_phi, 
                const std::vector<double>& v, 
                const double& M, 
                const std::unordered_map<int, std::vector<int>>& n, 
                const double& a, 
                const double& b, 
                const double& prop_sd);

double log_v_conditional(const std::vector<double>& v,
                         const std::vector<int>& n_j,
                         const std::unordered_map<int, std::vector<int>>& alpha,
                         double phi, 
                         double M);

std::vector<double> update_v(std::vector<double> curr_v,
                             const std::vector<double>& v_prop_sd,
                             double phi,
                             const std::unordered_map<int, std::vector<int>>& alpha,
                             const double& M);

Samples sample(const std::vector<std::vector<double>>& y_, 
               const int& n_samples, 
               const int& burn_in, 
               const std::vector<double>& params, 
               const std::vector<double>& y_pred_={},  
               const double& M_a=2.0, 
               const double& M_b=2.0, 
               const double& phi_a=2.0, 
               const double& phi_b=2.0, 
               const double& u_upper=0.1,
               const double& M_prop_sd=0.1, 
               const double& phi_prop_sd=0.3, 
               const bool& posterior_checks=false,
               const int& h = 20,
               const bool& verbose=true,
               const double& dx=1e-4);

#endif