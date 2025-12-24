#ifndef OPTION_PRICING_H
#define OPTION_PRICING_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <stdint.h>
#include <time.h>

#include "pcg_basic.h"

typedef struct {
    int num_sims;
    double S;
    double K;
    double r;
    double v;
    double T;
    pcg32_random_t rng;
    double call_payoff_sum;
    double put_payoff_sum;
} pricing_params_t;

double gaussian_box_muller(pcg32_random_t *rng);

double monte_carlo_call_payoff_sum(pricing_params_t* params);

double monte_carlo_put_payoff_sum(pricing_params_t* params);

void* thread_pricing_wrapper(void* arg);

#endif
