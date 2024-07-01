library(dplyr)
library(plotrix)
library(cmdstanr)


df <- read.csv("r_data/barley_data.csv.gz")
row.names(df) <- df[[1]]
df <- df[ , -1]

variances <- sapply(df, var, na.rm = TRUE)

# Step 2: Sort variances in descending order
sorted_variances <- sort(variances, decreasing = TRUE)

# Step 3: Calculate the number of columns to select (30% of total columns)
num_columns <- as.integer(length(sorted_variances) * 0.2)

# Step 4: Select the column names for the top 30% highest variance columns
top_variance_columns <- names(sorted_variances)[1:num_columns]

# Step 5: Subset the DataFrame to retain only these columns
df <- df[, top_variance_columns]

y = as.matrix(df)



run_nmf_model <- function(y, seed, K, dropout = 0.1) {
  set.seed(seed) # Setting the seed for any operation that involves randomness
  if (dropout > 0 && dropout < 1) {
    total_rows <- dim(y)[1]
    rows_to_keep <- sample(seq_len(total_rows), size = total_rows * (1 - dropout))
    y <- y[rows_to_keep,] # Apply dropout to y
  }
  print(dim(y))
  m <- mean(y)
  v <- sd(y)^2
  b <- (2 * K * m) / (v - m^2)
  a <- (2 * m^2) / (v - m^2)
  print(paste0("Data mean: ", m, " var: ", v))
  print(paste0("Estimate mean: ", K * a / b, " var: ", K^2 * (a * (a + 2)) / b^2))
  print(c(a, b))

  file3 <- "global_model_light.stan"
  mod3 <- cmdstan_model(stan_file = file3)

  nmf_data3 <- list(
    U = dim(y)[1], # Adjusted to reflect potential changes in y due to dropout
    I = dim(y)[2], # This might need to be adjusted if it also needs to reflect changes in the dataset size
    K = K,
    y = y,
    a_start = a,
    b_start = b
  )

  fit_variational3 <- mod3$variational(
    data = nmf_data3,
    seed = seed,
    algorithm = "meanfield",
    iter = 100000,
    grad_samples = 1,
    elbo_samples = 100,
    tol_rel_obj = 0.0001,
    eval_elbo = 100,
    draws = 1000,
    adapt_engaged = TRUE
  )

  theta_hat3 <- matrix(data = pull(fit_variational3$summary(variables = c("theta"), "median"), name = "median"),
                       byrow = FALSE,
                       ncol = K)

  color2D.matplot(x = theta_hat3, Hinton = TRUE) # NMF results visualization

  # Save the results and permutation to CSV
  results_filename <- paste0("r_barley/barley_dropout_seed_", seed, "_K_", K, "_dropout_", dropout, ".csv")
  write.csv(theta_hat3, file = results_filename)
  if (dropout > 0 && dropout < 1) {
      # Save the permutation
      perm_filename <- paste0("r_barley/permutation_leukemia_seed_", seed, "_K_", K, "_dropout_", dropout, ".csv")
      write.csv(rows_to_keep, file = perm_filename)
  }
}

seeds <- c(2345, 3456, 4567, 5678) # , 3456, 4567, 5678
for (seed in seeds) {
  run_nmf_model(y, seed, K = 6, dropout = 0.1)
}

