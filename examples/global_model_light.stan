
// Poisson-gamma NMF
//
// Y ~ Poisson(Theta * B) with bells and whistles
//
// Kucukelbir et al. (2015), Automatic Variational Inference in Stan
//
// https://papers.nips.cc/paper_files/paper/2015/hash/352fe25daf686bdb4edca223c921acea-Abstract.html
//
// Updated code based on Figure 8 in Supplemental material 
//
// 16 April 2024

data {
  int<lower=0> U; // Number of users
  int<lower=0> I; // Number of items
  int<lower=0> K; // Latent space dimension
  array[U, I] int<lower=0> y; // Data matrix
  real a_start;
  real b_start;
}

parameters {
  real<lower=0> sigma_a; // Global hyperparameters
  real<lower=0> sigma_b;
  vector<lower=0>[K] alpha_gamma; // Local hyperparameters
  vector<lower=0>[K] beta_gamma;
  array[U] vector<lower=0>[K] theta; // Features
  // array[I] positive_ordered[K] beta; // Coefficients
  array[I] vector<lower=0>[K] beta; // Coefficients
  real<lower=0> phi;
}

model {
  // Priors on global hyperparameters
  sigma_a ~ lognormal(0, 4); 
  sigma_b ~ lognormal(0, 4);
  
  // Priors on local hyperparameters
  alpha_gamma ~ normal(a_start, sigma_a);
  beta_gamma ~ normal(b_start, sigma_b);


  // Prior on theta
  for (k in 1:K){
    for (u in 1:U){
    theta[u, k] ~ gamma(alpha_gamma[k], beta_gamma[k]); // Shape-rate parametrisation of gamma distribution
    }
  }
  
  // Prior on beta
  for (i in 1:I){
    beta[i] ~ exponential(1);
  }
  
  // Likelihood
  for (u in 1:U){
    for (i in 1:I){
      y[u, i] ~ poisson(dot_product(theta[u], beta[i]));
    }
  }
}

