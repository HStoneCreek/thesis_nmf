library(cmdstanr)
library(dplyr)
library(plotrix) # Hinton diagram


df <- read.csv('r_data/leukemia_data.csv')

y = as.matrix(df)

y <- round(sqrt(y))

run_nmf_model <- function(y, seed, K, dropout = 0.1) {
  set.seed(seed) # Setting the seed for any operation that involves randomness
  if (dropout > 0 && dropout < 1) {
      total_rows <- dim(y)[1]
      rows_to_keep <- sample(seq_len(total_rows), size = total_rows * (1 - dropout))
      y <- y[rows_to_keep,]
  }
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
    I = 5000, # This might need to be adjusted if it also needs to reflect changes in the dataset size
    K = K,
    y = y,
    a_start = a,
    b_start =b
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
  results_filename <- paste0("r_leukemia/leukemia_dropout_seed_", seed, "_K_", K, "_dropout_", dropout, ".csv")
  write.csv(theta_hat3, file = results_filename)
  if (dropout > 0 && dropout < 1) {
  # Save the permutation
  perm_filename <- paste0("r_leukemia/permutation_leukemia_seed_", seed, "_K_", K, "_dropout_", dropout, ".csv")
  write.csv(rows_to_keep, file = perm_filename)
  }
  
  posterior_df <- as.data.frame(fit_variational3$draws())
  posterior_long_df <- posterior_df[, grep("^alpha_gamma\\[|^beta_gamma\\[", names(posterior_df))]
  
  for (i in 1:K) {
    old_alpha_col <- paste("alpha_gamma[", i, "]", sep = "")
    new_alpha_col <- paste("alpha [", i -1, "]", sep = "")
    old_beta_col <- paste("beta_gamma[", i, "]", sep = "")
    new_beta_col <- paste("beta [", i -1, "]", sep = "")
    ratio_col <- paste("ratio [", i - 1, "]", sep = "")
    
    # Rename columns
    posterior_long_df <- posterior_long_df %>%
      rename(!!new_alpha_col := !!old_alpha_col,
             !!new_beta_col := !!old_beta_col)
    
    # Compute the ratio a[i] / b[i] for each sample
    posterior_long_df[[ratio_col]] <- posterior_long_df[[new_alpha_col]] / posterior_long_df[[new_beta_col]]
    
  }
  
  # Assuming 'df' is your dataframe
  df_long <- posterior_long_df %>%
    pivot_longer(
      cols = everything(),  # Converts every column to long format
      names_to = "parameter",  # Column where the old column names will be stored
      values_to = "value"  # Column where the values will be stored
    )
  
  ratio_medians <- df_long %>%
    #filter(str_detect(parameter, "^ratio")) %>%
    group_by(parameter) %>%
    summarize(median_value = median(value))
  
  
  # Plotting
  # Your plotting code with improvements
  p <- ggplot(df_long, aes(x = value)) +
    geom_density(fill = "grey", alpha = 0.5) +
    facet_wrap(~parameter, scales = "free", ncol = K) +
    labs(title = "Posterior density per latent variable", x = "Value", y = "Density") +
    theme_minimal(base_family = "Arial") + 
    theme(
      panel.background = element_rect(fill = "white", colour = NA),
      plot.background = element_rect(fill = "white", colour = NA),
      strip.background = element_rect(fill = "white", colour = NA),
      strip.text = element_text(size = 14, face = "bold"),
      axis.text.x = element_text(size = 14, angle = 45, hjust = 1),
      axis.text.y = element_text(size = 14),
      axis.title.x = element_text(size = 18),  # Increase x-axis label font size
      axis.title.y = element_text(size = 18),  # Increase y-axis label font size
      plot.title = element_text(size = 20, face = "bold"),
      plot.margin = unit(c(1, 1, 1, 1), "cm"),
      #panel.grid.major = element_blank(),  # Remove major grid lines
      #panel.grid.minor = element_blank()   # Remove minor grid lines
    ) +
    scale_x_continuous(labels = scales::number_format(accuracy = 0.01))+
    scale_y_continuous(expand = expansion(mult = c(0, 0.2)))
  
  p <- p + geom_text(
    data = ratio_medians, 
    aes(label = paste0("Median: ", round(median_value, 2)), x = Inf, y = Inf), 
    hjust = 1, vjust = 2, size = 5, color = "black"
  )
  
  
  # Save the plot with high resolution and white background
  ggsave(filename = "figs/posterior_density_leukemia.png", plot = p, width = 20, height = 12, dpi = 300, bg = "white")
  
  
  return(fit_variational3)
}


# Example usage:
seeds <- c(2345, 3456, 4567, 5678, 6789, 2123)
for (seed in seeds) {
  model = run_nmf_model(y, seed, K =3, dropout = 0.1)
}





