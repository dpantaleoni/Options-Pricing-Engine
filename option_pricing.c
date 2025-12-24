/* Adapted from the C++ version of the Monte Carlo method for pricing options found
at https://www.quantstart.com/articles/European-vanilla-option-pricing-with-C-via-Monte-Carlo-methods/ */

#include "option_pricing.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b)) // Macro for max() fuunctionality 

// A simple implementation of the Box-Muller algorithm, used to generate
// gaussian random numbers - necessary for the Monte Carlo method below
double gaussian_box_muller(pcg32_random_t *rng) {
  double x = 0.0;
  double y = 0.0;
  double euclid_sq = 0.0;

  // Continue generating two uniform random variables
  // until the square of their "euclidean distance" 
  // is less than unity
  do {
    x = 2.0 * pcg32_random_r(rng) / (double)UINT32_MAX - 1;
    y = 2.0 * pcg32_random_r(rng) / (double)UINT32_MAX - 1;
    euclid_sq = x*x + y*y;
  } while (euclid_sq >= 1.0);

  return x*sqrt(-2*log(euclid_sq)/euclid_sq);
}

double monte_carlo_call_payoff_sum(pricing_params_t* params) {
  double S_adjust = params->S * exp(params->T*(params->r-0.5*params->v*params->v));
  double S_cur = 0.0;
  double payoff_sum = 0.0;

  for (int i=0; i<params->num_sims; i++) {
    double gauss_bm = gaussian_box_muller(&params->rng);
    S_cur = S_adjust * exp(sqrt(params->v*params->v*params->T)*gauss_bm);
    payoff_sum += MAX(S_cur - params->K, 0.0);
  }

  return payoff_sum;
}

double monte_carlo_put_payoff_sum(pricing_params_t* params) {
  double S_adjust = params->S * exp(params->T*(params->r-0.5*params->v*params->v));
  double S_cur = 0.0;
  double payoff_sum = 0.0;

  for (int i=0; i<params->num_sims; i++) {
    double gauss_bm = gaussian_box_muller(&params->rng);
    S_cur = S_adjust * exp(sqrt(params->v*params->v*params->T)*gauss_bm);
    payoff_sum += MAX(params->K - S_cur, 0.0);
  }

  return payoff_sum;
}

// wrapper function for the monte carlo call and put pricing functions
void* thread_pricing_wrapper(void* arg) {
  pricing_params_t* params = (pricing_params_t*)arg;
  params->call_payoff_sum = monte_carlo_call_payoff_sum(params);
  params->put_payoff_sum = monte_carlo_put_payoff_sum(params);
  return NULL;
}

int main(int argc, char* argv[]) {

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  int num_threads = (int)sysconf(_SC_NPROCESSORS_ONLN);
  uint64_t base_seed = (uint64_t)time(NULL);
  if (num_threads < 1) {
    num_threads = 1;
  }
  // First we create the parameter list                                                                               
  int num_sims = 10000000;   // Number of simulated asset paths
  num_sims = num_sims - (num_sims % num_threads); // remove any remainder from the number of simulations                                                   
  double S = 100.0;  // Option price                                                                                  
  double K = 100.0;  // Strike price                                                                                  
  double r = 0.05;   // Risk-free rate (5%)                                                                           
  double v = 0.2;    // Volatility of the underlying (20%)                                                            
  double T = 1.0;    // One year until expiry
  
  // each thread will calculate the call and put payoff sums for a portion of the simulations
  pthread_t threads[num_threads];
  pricing_params_t thread_params[num_threads];
  
  for (int i = 0; i < num_threads; i++) {
    thread_params[i] = (pricing_params_t){
      .num_sims = num_sims / num_threads, // portion of the simulations for each thread
      .S = S,
      .K = K,
      .r = r,
      .v = v,
      .T = T,
      .call_payoff_sum = 0.0,
      .put_payoff_sum = 0.0,
    };
    // Initialize PCG RNG with unique seed for each thread
    pcg32_srandom_r(&thread_params[i].rng, base_seed + (uint64_t)i, (uint64_t)i);
    pthread_create(&threads[i], NULL, thread_pricing_wrapper, (void*)&thread_params[i]);
  }

  for (int i = 0; i < num_threads; i++) {
    pthread_join(threads[i], NULL);
  }

  // Sum all payoff sums from all threads
  double total_call_payoff_sum = 0.0;
  double total_put_payoff_sum = 0.0;
  for (int i = 0; i < num_threads; i++) {
    total_call_payoff_sum += thread_params[i].call_payoff_sum;
    total_put_payoff_sum += thread_params[i].put_payoff_sum;
  }

  // Calculate final prices once from the aggregated payoff sums
  double call_price = (total_call_payoff_sum / (double)num_sims) * exp(-r*T);
  double put_price = (total_put_payoff_sum / (double)num_sims) * exp(-r*T);

  clock_gettime(CLOCK_MONOTONIC, &end);
  double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

  // Finally we output the parameters and prices and other useful information
  printf("Elapsed time: %.6f seconds\n", elapsed);
  // printf("Number of threads: %d\n", num_threads);                                                                      
  // printf("Number of Paths: %d\n", num_sims);
  // printf("Underlying:      %.2f\n", S);
  // printf("Strike:          %.2f\n", K);
  // printf("Risk-Free Rate:  %.2f\n", r);
  // printf("Volatility:      %.2f\n", v);
  // printf("Maturity:        %.2f\n", T);

  printf("Call Price:      %.6f\n", call_price);
  printf("Put Price:       %.6f\n", put_price);

  return 0;
}