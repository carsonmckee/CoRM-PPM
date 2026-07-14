#include "sncorm.h"
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>

double sampleVariance(const std::vector<double>& x){
    const std::size_t n = x.size();
    /*
    if (n < 2)
        throw std::invalid_argument("Need at least 2 observations");
    */

    double mean = std::accumulate(x.begin(), x.end(), 0.0) / n;

    double sumsq = 0.0;
    for (double v : x)
    {
        double d = v - mean;
        sumsq += d * d;
    }

    return sumsq / (n - 1); // unbiased sample variance
}

double acf(const std::vector<double>& x, int lag){
    const std::size_t n = x.size();

    /*
    if (lag < 0 || static_cast<std::size_t>(lag) >= n)
        throw std::invalid_argument("Invalid lag");
    */

    double mean = std::accumulate(x.begin(), x.end(), 0.0) / n;

    double denom = 0.0;
    for (double v : x)
    {
        double d = v - mean;
        denom += d * d;
    }

    if (denom == 0.0)
        return 0.0;

    double numer = 0.0;
    for (std::size_t t = lag; t < n; ++t)
    {
        numer += (x[t] - mean) * (x[t - lag] - mean);
    }

    return numer / denom;
}

std::mt19937& global_rng(){
    static std::mt19937 gen((unsigned int) time(NULL));
    return gen;
}

double sample_beta(const double& alpha, const double& beta) {
    std::gamma_distribution<double> dist_alpha(alpha, 1.0);
    std::gamma_distribution<double> dist_beta(beta, 1.0);

    double x = dist_alpha(global_rng());
    double y = dist_beta(global_rng());

    return x / (x + y);
}

void write_csv(const std::string& filename,
               const std::vector<std::vector<double>>& data) {
    
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << "\n";
        return;
    }

    for (const auto& row : data) {
        for (size_t j = 0; j < row.size(); ++j) {
            file << row[j];
            if (j < row.size() - 1) {
                file << ",";
            }
        }
        file << "\n";
    }

    file.close();
}

void write_csv(const std::string& filename,
               const std::vector<std::vector<int>>& data) {
    
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << "\n";
        return;
    }

    for (const auto& row : data) {
        for (size_t j = 0; j < row.size(); ++j) {
            file << row[j];
            if (j < row.size() - 1) {
                file << ",";
            }
        }
        file << "\n";
    }

    file.close();
}

int countUnique(const std::vector<int>& v) {
    std::unordered_set<int> s(v.begin(), v.end());
    return s.size();
}

std::vector<std::vector<double>> read_csv(const std::string& filename) {
    std::vector<std::vector<double>> data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << "\n";
        return data;
    }

    std::string line;

    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;

        while (std::getline(ss, cell, ',')) {
            try {
                row.push_back(std::stod(cell));
            } catch (...) {
                // Handle non-numeric values if needed
                row.push_back(0.0); // or skip / throw error
            }
        }

        if (!row.empty()) {
            data.push_back(row);
        }
    }

    return data;
}

double dnorm(const double& x, const double& mu, const double& sd, const bool& log){
    double val = -0.5*(x-mu)*(x-mu)/(sd*sd) - 0.5*std::log(2*sd*sd*PI);
    if(log){
        return val;
    }else{
        return std::exp(val);
    }
}

double dgamma(const double& x, const double& a, const double& b, const bool& log){
    double val = a*std::log(b) - std::lgamma(a) + (a-1)*std::log(x) - b*x;
    if (log){
        return val;
    }else{
        return std::exp(val);
    }
}

double rgamma(double a, double b){
    std::gamma_distribution<double> dist(a, 1.0 / b);
    return dist(global_rng());
}

std::pair<double, double> sample_posterior(
    double sum_yy,
    double sum_xy,
    double sum_xx,
    double n,
    double lam,
    double a,
    double b){
    // --- posterior hyperparameters ---
    double a_n = a + 0.5 * n;
    double lam_n = lam + sum_xx;

    double m_n = sum_xy / lam_n;

    double b_n = b + 0.5 * (sum_yy - m_n * m_n * lam_n);

    // --- safety checks (important in practice) ---
    if (lam_n <= 0.0) lam_n = 1e-12;
    if (b_n <= 0.0) b_n = 1e-12;
    
    // --- distributions ---
    // std::gamma_distribution<double> gamma_dist(a_n, 1.0 / b_n);
    // double gamma_sample = gamma_dist(global_rng());
    double gamma_sample = rgamma(a_n, b_n);
    
    double sigma2 = 1.0 / gamma_sample;
    
    std::normal_distribution<double> normal_dist(m_n, std::sqrt(sigma2 / lam_n));
    double phi = normal_dist(global_rng());
    
    return {phi, sigma2};
}

double log_g(const double& sum_yy,
             const double& sum_xy,
             const double& sum_xx, 
             const double& n,
             const std::vector<double>& params){
    double lam = params[0];
    double a = params[1];
    double b = params[2];
    if(n == 0){
        return 0;
    }
    double lgamma_a = std::lgamma(a);
    double log_b = std::log(b);
    double log_lam = std::log(lam);

    double a_n = a + 0.5*n;
    double lam_n = lam + sum_xx;
    double m_n = sum_xy / lam_n;
    double b_n = b + 0.5*(sum_yy - m_n*m_n * lam_n);
    
    double val = -0.5*n*log2pi + 0.5*log_lam - 0.5*std::log(lam_n);
    val += (a*log_b - lgamma_a - a_n*std::log(b_n) + std::lgamma(a_n));
    
    return val;
}

double log_g(const double& sum_yy,
             const double& sum_xy,
             const double& sum_xx, 
             const double& n,
             const double& lgamma_a, 
             const double& log_b, 
             const double& log_lam, 
             const std::vector<double>& params){
    
    double lam = params[0];
    double a = params[1];
    double b = params[2];

    if(n == 0){
        return 0;
    }

    double a_n = a + 0.5*n;
    double lam_n = lam + sum_xx;
    double m_n = sum_xy / lam_n;
    double b_n = b + 0.5*(sum_yy - m_n*m_n * lam_n);
    
    double val = -0.5*n*log2pi + 0.5*log_lam - 0.5*std::log(lam_n);
    val += (a*log_b - lgamma_a - a_n*std::log(b_n) + std::lgamma(a_n));
    
    return val;
}

double directing_intensity(const double& z, const double& phi, const double& M){
    return M * std::pow(1-z, phi-1) / z;
}

/*
double log_tau(const double& a, const double& z, const double& v, const double& phi){
    if(v == 0){
        return 0.0;
    }
    return std::lgamma(a + phi) - std::lgamma(phi) + (-a-phi)*std::log(1+v*z);
}

double kappa(const std::vector<int>& alpha, 
             const std::vector<double>& v_sums, 
             const double& phi, 
             const double& M, 
             double dx){
    const double z_start = 1e-20;
    const double z_end = 1.0 - 1e-20;
    const int I = (z_end - z_start) / dx;
    double integral_sum = 0.0;
    double sum_alpha = 0.0;
    
    for (double a : alpha) sum_alpha += a;
    
    // #pragma omp parallel for reduction(+:integral_sum)
    for (int zi = 0; zi < I; zi++) {
        double z = z_start + zi*dx;
        double log_val = sum_alpha * std::log(z) + std::log(directing_intensity(z, phi, M));
        
        for (size_t i = 0; i < alpha.size(); ++i) {
            log_val += log_tau(alpha[i], z, v_sums[i], phi);
        }

        integral_sum += std::exp(log_val);
    }

    return dx * integral_sum;
}
*/

double log_directing_intensity(
    const double& z,
    const double& phi,
    const double& M) {
    // log(M * (1-z)^(phi-1) / z)
    return std::log(M) + (phi - 1.0) * std::log1p(-z) - std::log(z);
}

double log_tau(
    const int& a,
    const double& z,
    const double& v,
    const double& phi) {
    if (v == 0.0) return 0.0;

    return std::lgamma(a + phi)
         - std::lgamma(phi)
         - (a + phi) * std::log1p(v * z);   // stable for small vz
}

double log_tau2(
    const int& a,
    const double& lgamma_a_phi, // lgamma(a + phi)
    const double& log1p_vz, // std::log1p(v * z)
    const double& lgamma_phi, // std::lgamma(phi)
    const double& phi) {
    
    return lgamma_a_phi - lgamma_phi - (a + phi) * log1p_vz; 
}

double kappa(
    const std::vector<int>& alpha,
    const std::vector<double>& v_sums,
    const double& phi,
    const double& M,
    double dx) {
    const double z_start = 1e-20;
    const double z_end   = 1.0 - 1e-20;
    const int I = static_cast<int>((z_end - z_start) / dx);
    
    double sum_alpha = 0.0;
    for (int a : alpha) sum_alpha += a;
    
    // First pass: compute log-integrand and max
    std::vector<double> log_vals(I);
    double max_log = -std::numeric_limits<double>::infinity();
    
    for (int zi = 0; zi < I; ++zi) {
        double z = z_start + zi * dx;

        double log_val =
            sum_alpha * std::log(z)
            + log_directing_intensity(z, phi, M);

        for (size_t i = 0; i < alpha.size(); ++i) {
            log_val += log_tau(alpha[i], z, v_sums[i], phi);
        }

        log_vals[zi] = log_val;
        max_log = std::max(max_log, log_val);
    }

    // Second pass: stabilized sum
    double scaled_sum = 0.0;

    for (double lv : log_vals) {
        scaled_sum += std::exp(lv - max_log);
    }

    // Final rescale
    double log_integral = std::log(dx) + max_log + std::log(scaled_sum);

    return log_integral;
}

// optimized version
/*
double kappa(
    const std::vector<int>& alpha,
    const std::vector<double>& v_sums,
    const double& phi,
    const double& lgamma_phi,
    const double& M,
    const std::vector<double>& z_vals, 
    const std::vector<double>& log_z_vals, 
    const std::vector<double>& log_directing, 
    const std::vector<std::vector<double>>& log1p_vz) {
    
    const double dx = z_vals[1] - z_vals[0];
    
    const int I = log_z_vals.size();
    
    double sum_alpha = 0.0;
    for (int a : alpha) sum_alpha += a;
    
    // First pass: compute log-integrand and max
    std::vector<double> log_vals(I);
    double max_log = -std::numeric_limits<double>::infinity();
    
    std::vector<double> lgamma_a_phi(alpha.size());
    for(int i=0; i<alpha.size(); i++){
        lgamma_a_phi[i] = std::lgamma(alpha[i] + phi);
    }

    // #pragma omp parallel for
    for (int zi = 0; zi < I; ++zi) {

        double log_val = sum_alpha*log_z_vals[zi] + log_directing[zi];

        for (size_t i = 0; i < alpha.size(); ++i) {
            log_val += log_tau2(alpha[i], lgamma_a_phi[i], log1p_vz[i][zi], lgamma_phi, phi);
        }
        
        log_vals[zi] = log_val;
        max_log = std::max(max_log, log_val);
    }

    // Second pass: stabilized sum
    double scaled_sum = 0.0;

    for (double lv : log_vals) {
        scaled_sum += std::exp(lv - max_log);
    }

    // Final rescale
    double log_integral = std::log(dx) + max_log + std::log(scaled_sum);

    return log_integral;
}
*/

double kappa(
    const std::vector<int>& alpha,
    const std::vector<double>& v_sums,
    const double& phi,
    const double& lgamma_phi,
    const double& M,
    const std::vector<double>& z_vals, 
    const std::vector<double>& log_z_vals, 
    const std::vector<double>& log_directing, 
    const std::vector<std::vector<double>>& log1p_vz) {
    
    const double dx = z_vals[1] - z_vals[0];
    const int I = log_z_vals.size();
    
    double sum_alpha = 0.0;
    for (int a : alpha)
        sum_alpha += a;
    
    // First pass: compute log-integrand and max
    std::vector<double> log_vals(I);

    double max_log = -std::numeric_limits<double>::infinity();
    
    std::vector<double> lgamma_a_phi(alpha.size());

    for (size_t i = 0; i < alpha.size(); i++) {
        lgamma_a_phi[i] = std::lgamma(alpha[i] + phi);
    }

    // Compute log integrand
    for (int zi = 0; zi < I; ++zi) {

        double log_val =
            sum_alpha * log_z_vals[zi]
            + log_directing[zi];

        for (size_t i = 0; i < alpha.size(); ++i) {

            log_val += log_tau2(
                alpha[i],
                lgamma_a_phi[i],
                log1p_vz[i][zi],
                lgamma_phi,
                phi
            );
        }
        
        log_vals[zi] = log_val;

        if (log_val > max_log)
            max_log = log_val;
    }

    // Second pass: trapezoidal rule in stabilized log-space
    double scaled_sum = 0.0;

    for (int zi = 0; zi < I; ++zi) {

        double weight = 1.0;

        // Trapezoidal endpoint weights
        if (zi == 0 || zi == I - 1){
            weight = 0.5;
        }
        scaled_sum += weight * std::exp(log_vals[zi] - max_log);
    }

    // Final stabilized log integral
    double log_integral =
        std::log(dx)
        + max_log
        + std::log(scaled_sum);
    
    return log_integral;
}

std::vector<int> sample_multinomial(const std::vector<double>& W, size_t N){
    const size_t K = W.size();

    // Build CDF
    std::vector<double> cdf(K);
    cdf[0] = W[0];
    for (size_t i = 1; i < K; ++i)
        cdf[i] = cdf[i - 1] + W[i];

    const double total = cdf.back();

    std::uniform_real_distribution<double> dist(0.0, total);

    std::vector<int> out;
    out.reserve(N);

    for (size_t i = 0; i < N; ++i) {
        double r = dist(global_rng());
        auto it = std::lower_bound(cdf.begin(), cdf.end(), r);
        out.push_back(int(it - cdf.begin()));
    }

    return out;
}

int multinomial_sample(const std::vector<double>& weights, const int& n_parts){
    if (weights.empty()) return -1; // safety check

    // Compute cumulative sum
    std::vector<double> cum_weights(n_parts);
    cum_weights[0] = weights[0];
    for (size_t i = 1; i < n_parts; ++i) {
        cum_weights[i] = cum_weights[i - 1] + weights[i];
    }
    
    // Sample uniform [0,1)
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    double u = uniform_dist(global_rng());

    // Find first index where u < cumulative sum
    for (size_t i = 0; i < n_parts; ++i) {
        if (u < cum_weights[i]) return static_cast<int>(i);
    }

    // Fallback (should rarely happen if weights sum to 1)
    return static_cast<int>(n_parts - 1);
}

double M_f_z_t(const double& z, const double& t, const double& phi){
    if(t == 0){
        return 1.0;
    }
    return std::pow(1-z*t, -phi);
}

double psi(const std::vector<double>& v, const double& phi, const double& M, double dx){
    const double z_start = 1e-20;
    const double z_end = 1.0 - 1e-20;

    double integral_sum = 0.0;

    // Loop over z values
    for (double z = z_start; z < z_end; z += dx) {
        double v_star = directing_intensity(z, phi, M);

        // Compute product over M_f_z_t
        double log_v = std::log(M_f_z_t(z, -v[0], phi));
        for (size_t j = 1; j < v.size(); ++j) {
            log_v += std::log(M_f_z_t(z, -v[j], phi));
        }

        double prod_M = std::exp(log_v);
        double integrand = (1.0 - prod_M) * v_star;

        integral_sum += integrand;
    }

    return dx * integral_sum;
}

void print_vec(const std::vector<double>& arr){
    for(int i=0; i<arr.size(); i++){
        std::cout << arr[i] << ", ";
    }
    std::cout << std::endl;
}

void print_vec(const std::vector<int>& arr){
    for(int i=0; i<arr.size(); i++){
        std::cout << arr[i] << ", ";
    }
    std::cout << std::endl;
}

double sum(std::vector<double>& arr){
    double s = 0.0;
    for(int i=0; i<arr.size(); i++){
        s += arr[i];
    }
    return s;
}

int sum(std::vector<int>& arr){
    int s = 0.0;
    for(int i=0; i<arr.size(); i++){
        s += arr[i];
    }
    return s;
}

double ar1_loglik(
    double Syy,
    double Sxy,
    double Sxx,
    int N,
    double phi,
    double sigma2){

    double quad =
        Syy
        - 2.0 * phi * Sxy
        + phi * phi * Sxx;

    double loglik =
        -0.5 * N * std::log(2.0 * PI * sigma2)
        -0.5 * quad / sigma2;

    return loglik;
}

int get_next_sp(const int& j, const int& t, const std::vector<std::vector<int>>& runs){
    int T = runs[0].size();
    int next_sp = T;
    for(int k=t+1; k<T; k++){
        if(runs[j][k] == 1){
            next_sp = k;
            break;
        }
    }
    return next_sp;
}

double compute_log_model_like(
    std::unordered_map<int, std::vector<double>>& state_summary_stats,
    const double& lgamma_a, 
    const double& log_b, 
    const double& log_lam, 
    const std::vector<double>& params){
    double val = 0.0;
    // #pragma omp parallel for reduction(+:val)
    for (auto& [key, value] : state_summary_stats){
        if(value[3] > 0){
            val += log_g(value[0], value[1], value[2], value[3], params);
        }
    }
    return val;
}

void sample_runs_states(
    const std::vector<std::vector<double>>& yy_sums, 
    const std::vector<std::vector<double>>& xy_sums,
    const std::vector<std::vector<double>>& xx_sums,
    const std::vector<double>& seg_n_sums,
    std::vector<std::vector<int>>& run_lengths,
    std::vector<std::vector<int>>& states,
    std::unordered_map<int, std::vector<double>>& state_summary_stats,
    std::unordered_set<int>& available_cluster_labels,
    std::unordered_map<int, std::vector<int>>& n,
    std::vector<std::vector<int>>& corm_clusters,
    const double& M, 
    const double& phi, 
    const std::vector<double>& curr_v,
    const std::vector<double>& params, 
    const std::vector<double>& cp_probs, 
    std::unordered_map<std::vector<int>, double, VectorHash>& kappa_cache, 
    const double& lgamma_phi,
    const std::vector<double>& z_vals, 
    const std::vector<double>& log_z_vals, 
    const std::vector<double>& log_directing, 
    const std::vector<std::vector<double>>& log1p_vz){
    
    int d = states.size();
    int T = states[0].size();
    
    int prev_sp, next_sp, curr_state, curr_cluster;
    double yy, xy, xx, seg_n, prob, log_like, num, denom;
    
    double lgamma_a = std::lgamma(params[1]);
    double log_b = std::log(params[2]);
    double log_lam = std::log(params[0]);
    
    for(int j=0; j<d; j++){
        prev_sp = 0;
        next_sp = get_next_sp(j, 0, run_lengths);
        std::vector<int> n_j(d, 0);
        n_j[j] = 1;

        // U_0 = 1 always so just need to compute K+1 values of corm (which in this case maps to state always)
        // seg_n = (double) (next_sp);
        seg_n = seg_n_sums[next_sp] - seg_n_sums[0];
        yy = yy_sums[j][next_sp] - yy_sums[j][0];
        xy = xy_sums[j][next_sp] - xy_sums[j][0];
        xx = xx_sums[j][next_sp] - xx_sums[j][0];

        curr_state = curr_cluster = states[j][0];
        
        state_summary_stats[curr_state][0] -= yy;
        state_summary_stats[curr_state][1] -= xy;
        state_summary_stats[curr_state][2] -= xx;
        state_summary_stats[curr_state][3] -= seg_n;

        if(sum(n[curr_cluster]) == 1){
            n.erase(curr_cluster);
            state_summary_stats.erase(curr_cluster);
            available_cluster_labels.insert(curr_cluster);
        } else {
            n[curr_cluster][j] -= 1;
        }
        std::vector<double> corm_labels;
        std::vector<double> log_corm_probs;

        for (auto& [key, value] : n) {
            value[j] += 1;
            if(kappa_cache.count(value) > 0){
                num = kappa_cache[value];
            } else {
                // num = kappa(value, curr_v, phi, M);
                num = kappa(value, curr_v, phi, lgamma_phi, M, z_vals, log_z_vals, log_directing, log1p_vz);
                kappa_cache[value] = num;
            }
            value[j] -= 1;
            if(kappa_cache.count(value) > 0){
                denom = kappa_cache[value];
            } else {
                // denom = kappa(value, curr_v, phi, M);
                denom = kappa(value, curr_v, phi, lgamma_phi, M, z_vals, log_z_vals, log_directing, log1p_vz);
                kappa_cache[value] = denom;
            }

            prob = num - denom;

            state_summary_stats[key][0] += yy;
            state_summary_stats[key][1] += xy;
            state_summary_stats[key][2] += xx;
            state_summary_stats[key][3] += seg_n;

            log_like = compute_log_model_like(state_summary_stats, lgamma_a, log_b, log_lam, params);
            
            // log_like = log_g(state_summary_stats[key][0], state_summary_stats[key][1], state_summary_stats[key][2], state_summary_stats[key][3], lgamma_a, log_b, log_lam, params);
            // log_like -= log_g(state_summary_stats[key][0]-yy, state_summary_stats[key][1]-xy, state_summary_stats[key][2]-xx, state_summary_stats[key][3]-seg_n, lgamma_a, log_b, log_lam, params);

            prob += log_like;

            state_summary_stats[key][0] -= yy;
            state_summary_stats[key][1] -= xy;
            state_summary_stats[key][2] -= xx;
            state_summary_stats[key][3] -= seg_n;

            corm_labels.push_back(key);
            log_corm_probs.push_back(prob);
        }
        // std::cout << "here4\n";

        int new_cluster = *available_cluster_labels.begin();
        double val;
        if(kappa_cache.count(n_j) > 0){
            val = kappa_cache[n_j];
        } else {
            // val = kappa(n_j, curr_v, phi, M);
            val = kappa(n_j, curr_v, phi, lgamma_phi, M, z_vals, log_z_vals, log_directing, log1p_vz);
            kappa_cache[n_j] = val;
        }
        prob = val;

        state_summary_stats[new_cluster] = {yy, xy, xx, seg_n};

        log_like = compute_log_model_like(state_summary_stats, lgamma_a, log_b, log_lam, params);
        // log_like = log_g(yy, xy, xx, seg_n, lgamma_a, log_b, log_lam, params);
        prob += log_like;

        state_summary_stats[new_cluster] = {0.0, 0.0, 0.0, 0.0};

        corm_labels.push_back(new_cluster);
        log_corm_probs.push_back(prob);
        available_cluster_labels.erase(new_cluster);
        n[new_cluster] = n_j;
        n[new_cluster][j] -= 1;

        // subtract max log, undo log and normalise
        double mx = *std::max_element(log_corm_probs.begin(), log_corm_probs.end());
        double prob_sum = 0.0;
        for(int k=0; k<log_corm_probs.size(); k++){
            log_corm_probs[k] -= mx;
            log_corm_probs[k] = std::exp(log_corm_probs[k]);
            prob_sum += log_corm_probs[k];
        }
        
        for(int k=0; k<log_corm_probs.size(); k++){
            log_corm_probs[k] /= prob_sum;
        }
        
        // sample 
        int ind = multinomial_sample(log_corm_probs, log_corm_probs.size());
        int new_value = corm_labels[ind];
        if(new_value != new_cluster){
            state_summary_stats.erase(new_cluster);
            available_cluster_labels.insert(new_cluster);
            n.erase(new_cluster);
        } 

        state_summary_stats[new_value][0] += yy;
        state_summary_stats[new_value][1] += xy;
        state_summary_stats[new_value][2] += xx;
        state_summary_stats[new_value][3] += seg_n;
        n[new_value][j] += 1;
        corm_clusters[j][0] = new_value;

        for(int k=0; k<next_sp; k++){
            states[j][k] = new_value;
        }

        for(int t=1; t<T; t++){
            if(run_lengths[j][t] == 1){
                next_sp = get_next_sp(j, t, run_lengths);
            }

            // 2(K+1) outcomes
            
            // start by removing current values from the corm and summary stats
            // seg_n = (double) (next_sp - t);
            seg_n = seg_n_sums[next_sp] - seg_n_sums[t];
            yy = yy_sums[j][next_sp] - yy_sums[j][t];
            xy = xy_sums[j][next_sp] - xy_sums[j][t];
            xx = xx_sums[j][next_sp] - xx_sums[j][t];

            curr_state = states[j][t];
            curr_cluster = corm_clusters[j][t];
            
            state_summary_stats[curr_state][0] -= yy;
            state_summary_stats[curr_state][1] -= xy;
            state_summary_stats[curr_state][2] -= xx;
            state_summary_stats[curr_state][3] -= seg_n;
            
            // corm
            if(sum(n[curr_cluster]) == 1){
                n.erase(curr_cluster);
                state_summary_stats.erase(curr_cluster);
                available_cluster_labels.insert(curr_cluster);
            } else {
                n[curr_cluster][j] -= 1;
            }

            // compute corm probs
            std::vector<double> corm_labels;
            std::vector<double> log_corm_probs;
            for (auto& [key, value] : n) {
                value[j] += 1;
                if(kappa_cache.count(value) > 0){
                    num = kappa_cache[value];
                } else {
                    // num = kappa(value, curr_v, phi, M);
                    num = kappa(value, curr_v, phi, lgamma_phi, M, z_vals, log_z_vals, log_directing, log1p_vz);
                    kappa_cache[value] = num;
                }
                value[j] -= 1;
                if(kappa_cache.count(value) > 0){
                    denom = kappa_cache[value];
                } else {
                    // denom = kappa(value, curr_v, phi, M);
                    denom = kappa(value, curr_v, phi, lgamma_phi, M, z_vals, log_z_vals, log_directing, log1p_vz);
                    kappa_cache[value] = denom;
                }

                prob = num - denom;
                corm_labels.push_back(key);
                log_corm_probs.push_back(prob);
            }

            int new_cluster = *available_cluster_labels.begin();
            double val;
            if(kappa_cache.count(n_j) > 0){
                val = kappa_cache[n_j];
            } else {
                // val = kappa(n_j, curr_v, phi, M);
                val = kappa(n_j, curr_v, phi, lgamma_phi, M, z_vals, log_z_vals, log_directing, log1p_vz);
                kappa_cache[n_j] = val;
            }
            prob = val;
            corm_labels.push_back(new_cluster);
            log_corm_probs.push_back(prob);

            state_summary_stats[new_cluster] = {0.0, 0.0, 0.0, 0.0};
            available_cluster_labels.erase(new_cluster);
            n[new_cluster] = n_j;
            n[new_cluster][j] -= 1;
            
            std::vector<double> probs(2*log_corm_probs.size());
            
            // compute Ut=0 probs
            // state becomes equal to states[j][t-1]
            state_summary_stats[states[j][t-1]][0] += yy;
            state_summary_stats[states[j][t-1]][1] += xy;
            state_summary_stats[states[j][t-1]][2] += xx;
            state_summary_stats[states[j][t-1]][3] += seg_n;

            log_like = compute_log_model_like(state_summary_stats, lgamma_a, log_b, log_lam, params);
            // log_like = log_g(state_summary_stats[states[j][t-1]][0], state_summary_stats[states[j][t-1]][1], state_summary_stats[states[j][t-1]][2], state_summary_stats[states[j][t-1]][3], lgamma_a, log_b, log_lam, params);
            // log_like -= log_g(state_summary_stats[states[j][t-1]][0]-yy, state_summary_stats[states[j][t-1]][1]-xy, state_summary_stats[states[j][t-1]][2]-xx, state_summary_stats[states[j][t-1]][3]-seg_n, lgamma_a, log_b, log_lam, params);

            state_summary_stats[states[j][t-1]][0] -= yy;
            state_summary_stats[states[j][t-1]][1] -= xy;
            state_summary_stats[states[j][t-1]][2] -= xx;
            state_summary_stats[states[j][t-1]][3] -= seg_n;

            for(int k=0; k<log_corm_probs.size(); k++){
                probs[k] = std::log(1-cp_probs[j]) + log_like + log_corm_probs[k];
            }
            
            // compute Ut=1 probs
            // state becomes equal to corm_label[k]
            for(int k=0; k<log_corm_probs.size(); k++){
                // print_vec(state_summary_stats[corm_labels[k]]);
                state_summary_stats[corm_labels[k]][0] += yy;
                state_summary_stats[corm_labels[k]][1] += xy;
                state_summary_stats[corm_labels[k]][2] += xx;
                state_summary_stats[corm_labels[k]][3] += seg_n;
                
                log_like = compute_log_model_like(state_summary_stats, lgamma_a, log_b, log_lam, params);
                // log_like = log_g(state_summary_stats[corm_labels[k]][0], state_summary_stats[corm_labels[k]][1], state_summary_stats[corm_labels[k]][2], state_summary_stats[corm_labels[k]][3], lgamma_a, log_b, log_lam, params);
                // log_like -= log_g(state_summary_stats[corm_labels[k]][0]-yy, state_summary_stats[corm_labels[k]][1]-xy, state_summary_stats[corm_labels[k]][2]-xx, state_summary_stats[corm_labels[k]][3]-seg_n, lgamma_a, log_b, log_lam, params);

                state_summary_stats[corm_labels[k]][0] -= yy;
                state_summary_stats[corm_labels[k]][1] -= xy;
                state_summary_stats[corm_labels[k]][2] -= xx;
                state_summary_stats[corm_labels[k]][3] -= seg_n;

                probs[k + log_corm_probs.size()] = std::log(cp_probs[j]) + log_like + log_corm_probs[k];

            }

            // subtract max log, undo log and normalise
            double mx = *std::max_element(probs.begin(), probs.end());
            // std::cout << "max = " << mx << std::endl;
            double prob_sum = 0.0;
            for(int k=0; k<probs.size(); k++){
                probs[k] -= mx;
                probs[k] = std::exp(probs[k]);
                prob_sum += probs[k];
            }
            
            for(int k=0; k<probs.size(); k++){
                probs[k] /= prob_sum;
            }

            // sample 
            int ind = multinomial_sample(probs, probs.size());
            
            // update state, corm values and summary states
            // make sure to propagate to the future
            int new_U, new_state, new_c;
            if(ind < log_corm_probs.size()){
                new_U = 0;
                new_state = states[j][t-1];
                new_c = corm_labels[ind]; 
            } else {
                new_U = 1;
                new_state = corm_labels[ind - log_corm_probs.size()];
                new_c = corm_labels[ind - log_corm_probs.size()];
            }

            state_summary_stats[new_state][0] += yy;
            state_summary_stats[new_state][1] += xy;
            state_summary_stats[new_state][2] += xx;
            state_summary_stats[new_state][3] += seg_n;

            run_lengths[j][t] = new_U;
            corm_clusters[j][t] = new_c;
            n[new_c][j] += 1;

            if(new_c != new_cluster){
                n.erase(new_cluster);
                state_summary_stats.erase(new_cluster);
                available_cluster_labels.insert(new_cluster);
            }

            for(int k=t; k<next_sp; k++){
                states[j][k] = new_state;
            }

            if(new_U == 1){
                prev_sp = t;
            }

        }
    }
    
}

std::vector<int> flatten(const std::vector<std::vector<int>>& arr, const int& T){
    std::vector<int> out;
    int d = arr.size();

    for(int j=0; j<d; j++){
        for(int t=0; t<T; t++){
            out.push_back(arr[j][t]);
        }
    }
    return out;
}

std::vector<int> order_of_appearance(const std::vector<int>& arr){
    std::vector<int> out;

    int val;

    std::unordered_map<int, int> map;
    int counter = 0;

    for(int t=0; t<arr.size(); t++){
        val = arr[t];
        if(map.count(val) == 0){
            map[val] = counter;
            counter += 1;  
        } 
        out.push_back(map[val]);
    }
    return out;
}

double update_M(double& curr_M, 
                const std::vector<double>& v, 
                const double& phi, 
                const std::unordered_map<int, std::vector<int>>& n, 
                const double& a, 
                const double& b, 
                const double& prop_sd){
    
    double a_post = a + n.size();
    double b_post = b + psi(v, phi, 1);
    return rgamma(a_post, b_post);
}

double update_phi(double& curr_phi, 
                  const std::vector<double>& v, 
                  const double& M, 
                  const std::unordered_map<int, std::vector<int>>& n, 
                  const double& a, 
                  const double& b, 
                  const double& prop_sd
                ){
    
    int thin = 6;
    std::normal_distribution<double> normal_dist(0, prop_sd);
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    for(int th=0; th<thin; th++){
        double prop_phi = curr_phi + normal_dist(global_rng());
        if(prop_phi < 0){
            continue;
        }
        double log_prob = dgamma(prop_phi, a, b, true) - dgamma(curr_phi, a, b, true);
        log_prob += (-psi(v, prop_phi, M) + psi(v, curr_phi, M));
        for (auto& [key, value] : n){
            log_prob += (kappa(value, v, prop_phi, M) - kappa(value, v, curr_phi, M));
        }
        
        double prob = std::exp(log_prob);

        if(uniform_dist(global_rng()) < prob){
            curr_phi = prop_phi;
        }
    }
    return curr_phi;
}

double log_v_conditional(const std::vector<double>& v,
                         const std::vector<int>& n_j,
                         const std::unordered_map<int, std::vector<int>>& alpha,
                         double phi, 
                         double M){
    double val = 0.0;
    
    for (size_t i = 0; i < v.size(); ++i) {
        val += (n_j[i] - 1.0) * std::log(v[i]);
    }
    val -= psi(v, phi, M);
    
    for(auto& [key, value]: alpha){
        val += kappa(value, v, phi, M);
    }

    return val;
}

std::vector<double> update_v(std::vector<double> curr_v,
                             const std::vector<double>& v_prop_sd,
                             double phi,
                             const std::unordered_map<int, std::vector<int>>& alpha,
                             const double& M){
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    
    std::vector<int> n_j(curr_v.size(), 0);
    for(auto& [key, value]: alpha){
        for(int j=0; j<n_j.size(); j++){
            n_j[j] += value[j];
        }
    }

    // print_vec(n_j);
    for (size_t i = 0; i < curr_v.size(); ++i) {
        for (int it = 0; it < 6; ++it) { // thinning loop
            std::normal_distribution<double> normal_dist(curr_v[i], v_prop_sd[i]);
            double prop_v = normal_dist(global_rng());
            
            if (prop_v <= 0.0) continue;
            
            std::vector<double> proposal = curr_v;
            proposal[i] = prop_v;

            double log_accept_prob = log_v_conditional(proposal, n_j, alpha, phi, M) - log_v_conditional(curr_v, n_j, alpha, phi, M);
            
            double accept_prob = std::min(1.0, std::exp(log_accept_prob));

            if (uniform_dist(global_rng()) < accept_prob) {
                curr_v[i] = prop_v;
            }
        }
    }

    return curr_v;
}

Samples sample(const std::vector<std::vector<double>>& y_, 
               const int& n_samples, 
               const int& burn_in, 
               const std::vector<double>& params, 
               const std::vector<double>& y_pred_, 
               const double& M_a, 
               const double& M_b, 
               const double& phi_a, 
               const double& phi_b, 
               const double& u_upper,
               const double& M_prop_sd, 
               const double& phi_prop_sd, 
               const bool& posterior_checks,
               const int& h,
               const bool& verbose,
               const double& dx){
    
    Samples sample_output;
    
    std::vector<std::vector<double>> y, x;
    
    for(int j=0; j<y_.size(); j++){
        std::vector<double> temp1, temp2;
        x.push_back(temp1);
        y.push_back(temp2);
        
        for(int t = 0; t < y_[0].size()-1; t++){
            x[j].push_back(y_[j][t]);
            y[j].push_back(y_[j][t+1]);
        }
    }
    
    int d = y.size();
    int T = y[0].size();
    
    std::vector<double> y_pred;
    if(y_pred_.size() == 0){
        for(int j=0; j<d; j++){
            y_pred.push_back(0.0);
        }
    } else {
        y_pred = y_pred_;
    }

    std::vector<std::vector<double>> yy_sums;
    std::vector<std::vector<double>> xy_sums;
    std::vector<std::vector<double>> xx_sums;
    std::vector<double> seg_n_sums = {0};
    for(int t=0; t<T+h; t++){
        if(t < T){
            seg_n_sums.push_back(seg_n_sums[seg_n_sums.size()-1] + 1);
        } else{
            seg_n_sums.push_back(seg_n_sums[seg_n_sums.size()-1] + 0);
        }
    }

    for(int j=0; j<d; j++){
        std::vector<double> yy_sums_ = {0.0};
        std::vector<double> xy_sums_ = {0.0};
        std::vector<double> xx_sums_ = {0.0};
        for(int t=0; t<T+h; t++){
            if(t < T){
                yy_sums_.push_back(yy_sums_[yy_sums_.size()-1] + y[j][t]*y[j][t]);
                xy_sums_.push_back(xy_sums_[xy_sums_.size()-1] + x[j][t]*y[j][t]);
                xx_sums_.push_back(xx_sums_[xx_sums_.size()-1] + x[j][t]*x[j][t]);
            } else {
                yy_sums_.push_back(yy_sums_[yy_sums_.size()-1] + 0);
                xy_sums_.push_back(xy_sums_[xy_sums_.size()-1] + 0);
                xx_sums_.push_back(xx_sums_[xx_sums_.size()-1] + 0);
            }
        }
        yy_sums.push_back(yy_sums_);
        xy_sums.push_back(xy_sums_);
        xx_sums.push_back(xx_sums_);
    }
    
    // current values 
    double M = M_a / M_b;
    double phi = phi_a / phi_b;
    std::vector<double> cp_probs(d, 0.5*u_upper);
    std::vector<double> curr_v(d, 50.0);

    std::vector<double> v_prop_sd(d, 5.0);

    std::unordered_map<int, std::vector<int>> n;
    std::unordered_set<int> available_cluster_labels;

    std::vector<std::vector<int>> run_lengths;
    std::vector<std::vector<int>> corm_clusters;
    std::vector<std::vector<int>> states;

    std::unordered_map<int, std::vector<double>> state_summary_stats;

    for (int i = 0; i < d * (T+h); i++) {
        available_cluster_labels.insert(i);
    }
    
    // initialise 
    int counter = 0;
    for(int j=0; j<d; j++){
        run_lengths.push_back(std::vector<int>(T+h));
        corm_clusters.push_back(std::vector<int>(T+h));
        states.push_back(std::vector<int>(T+h));
        for(int t=0; t<T+h; t++){
            if(t == 0){
                std::vector<int> n_j(d, 0);
                n_j[j] = 1;
                n[counter] = n_j;
                run_lengths[j][t] = 1;
                states[j][t] = counter;
                state_summary_stats[counter] = {y[j][t]*y[j][t], y[j][t]*x[j][t], x[j][t]*x[j][t], 1.0};
                
            } else {
                n[counter][j] += 1;
                run_lengths[j][t] = run_lengths[j][t-1]+1;
                states[j][t] = states[j][t-1];
                if(t < T){
                    state_summary_stats[states[j][t]][0] += y[j][t]*y[j][t];
                    state_summary_stats[states[j][t]][1] += x[j][t]*y[j][t];
                    state_summary_stats[states[j][t]][2] += x[j][t]*x[j][t];
                    state_summary_stats[states[j][t]][3] += 1.0;
                }
            }
            corm_clusters[j][t] = counter;
        }
        available_cluster_labels.erase(counter);
        counter += 1;
    }

    std::vector<std::vector<int>> flat_run_store;
    std::vector<std::vector<int>> flat_states_store;
    std::vector<double> M_store;
    std::vector<double> phi_store;
    std::vector<int> n_active_states;
    
    // integral storage
    std::vector<double> z_vals;
    const double z_start = 1e-50;
    const double z_end   = 1.0 - 1e-50;
    int I = (int) (z_end - z_start) / dx;
    for(int i=0; i<I; i++){
        z_vals.push_back(z_start + i*dx);
    }
    
    std::vector<double> log_z_vals(I);
    std::vector<double> log_directing(I);
    std::vector<std::vector<double>> log1p_vz;
    for(int j=0; j<d; j++){
        log1p_vz.push_back(std::vector<double>(I));
    }

    double pred_est = 0.0;
    
    std::normal_distribution<double> STD_NORMAL(0, 1);
    
    std::vector<double> likelihood_sums(d*(T), 0.0);
    std::vector<double> log_likelihood_sums(d*(T), 0.0);
    std::vector<double> log_likelihood_2_sums(d*(T), 0.0);

    std::vector<std::vector<std::vector<double>>> residuals(d);
    std::vector<std::vector<double>> acf1s(d);
    std::vector<std::vector<double>> acf2s(d);
    std::vector<std::vector<double>> variances(d);
    
    for(int it=0; it<n_samples; it++){

        std::unordered_map<std::vector<int>, double, VectorHash> kappa_cache;
        
        // update interal caches
        double lgamma_phi = std::lgamma(phi);
        for(int i=0; i<I; i++){
            log_directing[i] = log_directing_intensity(z_vals[i], phi, M);
            log_z_vals[i] = std::log(z_vals[i]);
            for(int j=0; j<d; j++){
                log1p_vz[j][i] = std::log1p(curr_v[j]*z_vals[i]);
            }
        }

        sample_runs_states(
            yy_sums, 
            xy_sums,
            xx_sums,
            seg_n_sums,
            run_lengths,
            states,
            state_summary_stats,
            available_cluster_labels,
            n,
            corm_clusters,
            M, 
            phi, 
            curr_v,
            params, 
            cp_probs, 
            kappa_cache,
            lgamma_phi, 
            z_vals,
            log_z_vals, 
            log_directing,
            log1p_vz
        );

        // sample M
        M = update_M(M, 
                     curr_v, 
                     phi, 
                     n, 
                     M_a, 
                     M_b,
                     M_prop_sd);

        // update phi
        phi = update_phi(phi, 
                     curr_v, 
                     M, 
                     n, 
                     M_a, 
                     M_b,
                     M_prop_sd);

        // update_v 
        curr_v = update_v(curr_v, 
                          v_prop_sd,
                          phi, 
                          n, 
                          M);
        
        // update cp probs
        for(int j=0; j<d; j++){
            int N = sum(run_lengths[j]) - 1;
            double val = sample_beta(N + 1, T+h - N);
            while(val > u_upper){
                val = sample_beta(N + 1, T+h - N);
            }
            cp_probs[j] = val;
        }

        std::unordered_set<int> unique;
        for(int j=0; j<d; j++){
            for(int t=0; t<T; t++){
                unique.insert(states[j][t]);
            }
        }
        
        // sample parameters
        std::unordered_map<int, std::pair<double, double>> thetas;
        for(auto& [key, val]: state_summary_stats){
            thetas[key] = sample_posterior(val[0], val[1], val[2], val[3], params[0], params[1], params[2]);
        }

        double pred = 1.0;  
        for(int j=0; j<d; j++){
            pred *= dnorm(y_pred[j], thetas[states[j][T]].first * y[j][T-1], std::sqrt(thetas[states[j][T]].second));
        }

        if(verbose){
            std::cout << "Iter: " << it << ", num corm states: " << n.size() << ", num active states: " << unique.size() << ", M: " << M << ", phi: " << phi << ", v: ";
            for(int j=0; j<d; j++){
                std::cout << curr_v[j] << ", ";
            }
            std::cout << "pred_est: "<< pred << std::endl;
            // std::cout << std::endl;
        }

        if(it >= burn_in){
            std::vector<int> run_flat = flatten(run_lengths, T);
            std::vector<int> state_flat = flatten(states, T);
            flat_run_store.push_back(run_flat);
            flat_states_store.push_back(order_of_appearance(state_flat));
            M_store.push_back(M);
            phi_store.push_back(phi);
            n_active_states.push_back(unique.size());
            
            pred_est += pred;
            
            // WAIC
            for(int j=0; j<d; j++){
                for(int t=0; t<T; t++){
                    int i = j*(T) + t;
                    int s = states[j][t];
                    std::pair<double, double> theta = thetas[s];
                    double like = dnorm(y[j][t], theta.first * x[j][t], std::sqrt(theta.second));
                    
                    likelihood_sums[i] += like;
                    log_likelihood_sums[i] += std::log(like);
                    log_likelihood_2_sums[i] += std::log(like)*std::log(like);
                }
            }

            // residuals 
            if(posterior_checks){
                
                // residuals
                for(int j=0; j<d; j++){
                    std::vector<double> residuals_j;
                    for(int t=0; t<T; t++){
                        double phi_ = thetas[states[j][t]].first;
                        double sigma2_ = thetas[states[j][t]].second;
                        residuals_j.push_back((y[j][t] - phi_*x[j][t]) / std::sqrt(sigma2_));
                    }
                    residuals[j].push_back(residuals_j);
                }

                // posterior predictive
                for(int j=0; j<d; j++){
                    double prev_y = y[j][T-1];
                    std::vector<double> sim_y;
                    for(int h_=0; h_<h; h_++){
                        double phi_ = thetas[states[j][T+h_]].first;
                        double sigma2_ = thetas[states[j][T+h_]].second;
                        double new_y = phi_*prev_y + STD_NORMAL(global_rng()) * std::sqrt(sigma2_);
                        sim_y.push_back(new_y);
                        prev_y = new_y;
                    }
                    
                    // compute statistics
                    if(sim_y.size() > 2){
                        double acf1 = acf(sim_y, 1);
                        double acf2 = acf(sim_y, 2);
                        double variance = sampleVariance(sim_y);
                        
                        acf1s[j].push_back(acf1);
                        acf2s[j].push_back(acf2);
                        variances[j].push_back(variance);
                    }

                }

            }

        }
    }

    Samples samps;
    samps.states = flat_states_store;
    samps.runs = flat_run_store;
    samps.M = M_store;
    samps.phi = phi_store;
    samps.n_states = n_active_states;
    samps.y_pred = pred_est / (n_samples - burn_in);
    samps.residuals = residuals;
    samps.acf1s = acf1s;
    samps.acf2s = acf2s;
    samps.variances = variances;
    
    double lppd = 0.0;
    double penalty = 0.0;
    for(int j=0; j<d; j++){
        for(int t=0; t<T; t++){
            int i = j*T + t;
            lppd += std::log(likelihood_sums[i] / (n_samples - burn_in));
            penalty += (log_likelihood_2_sums[i] - log_likelihood_sums[i]*log_likelihood_sums[i]/(n_samples-burn_in)) / ((n_samples-burn_in) - 1);
        }
    }
    
    std::cout << "lppd = " << lppd << std::endl;
    std::cout << "waic = " << -2*(lppd - penalty) << std::endl;
    return samps;
}

/*
// g++ -Ofast -fopenmp -funroll-loops sncorm.cpp -o sncorm.exe
int main(){
    std::cout << "hello\n";

    double M   = 2.5;
    double phi = 1.7;

    // -----------------------------
    // inputs (same as before)
    // -----------------------------
    std::vector<int> a = {
        2,
        3,
        10
    };

    std::vector<double> v = {
        0.5,
        1.2,
        5
    };

    std::cout << kappa(a, v, phi, M) << std::endl;
}
*/
