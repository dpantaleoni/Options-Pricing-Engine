"""
Adapted from the C++ version of the Monte Carlo method for pricing options found
at https://www.quantstart.com/articles/European-vanilla-option-pricing-with-C-via-Monte-Carlo-methods/
"""

import random
import math
import time
import multiprocessing as mp

# A simple implementation of the Box-Muller algorithm, used to generate
# gaussian random numbers - necessary for the Monte Carlo method below.
def gaussian_box_muller(rng):
    x = 0.0
    y = 0.0
    euclid_sq = 0.0
    
    # Continue generating two uniform random variables
    # until the square of their "euclidean distance" 
    # is less than unity
    while True:
        x = 2.0 * rng.random() - 1
        y = 2.0 * rng.random() - 1
        euclid_sq = x * x + y * y
        if euclid_sq < 1.0:
            break
    
    return x * math.sqrt(-2 * math.log(euclid_sq) / euclid_sq)

# function to calculate the call option payoff sum using Monte Carlo simulation
def monte_carlo_call_payoff_sum(num_sims, S, K, r, v, T, seed):
    rng = random.Random(seed)
    
    S_adjust = S * math.exp(T * (r - 0.5 * v * v))
    payoff_sum = 0.0
    
    for i in range(num_sims):
        gauss_bm = gaussian_box_muller(rng)
        S_cur = S_adjust * math.exp(math.sqrt(v * v * T) * gauss_bm)
        payoff_sum += max(S_cur - K, 0.0)
    
    return payoff_sum

# function to calculate the put option payoff sum using Monte Carlo simulation
def monte_carlo_put_payoff_sum(num_sims, S, K, r, v, T, seed):
    rng = random.Random(seed)
    
    S_adjust = S * math.exp(T * (r - 0.5 * v * v))
    payoff_sum = 0.0
    
    for i in range(num_sims):
        gauss_bm = gaussian_box_muller(rng)
        S_cur = S_adjust * math.exp(math.sqrt(v * v * T) * gauss_bm)
        payoff_sum += max(K - S_cur, 0.0)
    
    return payoff_sum


def process_pricing_wrapper(args):
    num_sims, S, K, r, v, T, seed = args
    call_payoff_sum = monte_carlo_call_payoff_sum(num_sims, S, K, r, v, T, seed)
    put_payoff_sum = monte_carlo_put_payoff_sum(num_sims, S, K, r, v, T, seed)
    return call_payoff_sum, put_payoff_sum


def main():
    start_time = time.time()
    
    # Get number of CPU cores
    num_procs = mp.cpu_count()
    if num_procs < 1:
        num_procs = 1
    
    base_seed = int(time.time())
    
    # Parameters
    num_sims = 10000000  # Number of simulated asset paths
    num_sims = num_sims - (num_sims % num_procs)  # Remove remainder
    
    S = 100.0  # Option price (underlying)
    K = 100.0  # Strike price
    r = 0.05   # Risk-free rate (5%)
    v = 0.2    # Volatility of the underlying (20%)
    T = 1.0    # One year until expiry
    
    # Prepare arguments for each process
    sims_per_proc = num_sims // num_procs
    process_args = []
    for i in range(num_procs):
        seed = base_seed + i
        process_args.append((sims_per_proc, S, K, r, v, T, seed))
    
    # Run simulations in parallel
    with mp.Pool(processes=num_procs) as pool:
        results = pool.map(process_pricing_wrapper, process_args)
    
    # Sum all payoff sums from all processes
    total_call_payoff_sum = sum(result[0] for result in results)
    total_put_payoff_sum = sum(result[1] for result in results)
    
    # Calculate final prices once from the aggregated payoff sums
    call_price = (total_call_payoff_sum / num_sims) * math.exp(-r * T)
    put_price = (total_put_payoff_sum / num_sims) * math.exp(-r * T)
    
    elapsed = time.time() - start_time
    
    # Output results
    print("Elapsed time: {:.6f} seconds".format(elapsed))
    # print("Number of processes: {}".format(num_procs))
    # print("Number of Paths: {}".format(num_sims))
    # print("Underlying:      {:.2f}".format(S))
    # print("Strike:          {:.2f}".format(K))
    # print("Risk-Free Rate:  {:.2f}".format(r))
    # print("Volatility:      {:.2f}".format(v))
    # print("Maturity:        {:.2f}".format(T))
    print("Call Price:      {:.6f}".format(call_price))
    print("Put Price:       {:.6f}".format(put_price))


if __name__ == "__main__":
    main()

