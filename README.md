# thesis_nmf
This repository collects all the code used in my MSc thesis on matrix factorization techniques for dimensionality reduction and clustering.

## Datasets
We used three different datasets for our comparative study.

**Key information:**
- **Synthetic:** Located in `/synthetic_data`, data generation function is present in `synthetic_uniform.py`.
- **Leukemia:** Located in `/leukemia_data`, data has been retrieved from [this link](https://github.com/mims-harvard/nimfa/tree/master/nimfa/datasets/ALL_AML).
- **Barley:** Located in `/barley`, meta information has been retrieved from [this DOI](https://doi.org/10.1111/tpj.14414) and SNP marker data has been provided by Jonathan Kunst.

## Code Organization

### genolytics
PCA and (penalized) NMF benchmarking has been implemented in the `genolytics` package. 

Specifically:
- **models:** Implements the `PCCLustering` and `NMFmodel` which allow for easy execution of dimension reduction, followed by k-means clustering.
- **sqlalchemy:** Used to keep track of different runs.
- **genolytics.benchmark:** Implements the logic to execute a parameterized run and write the result into a SQLite database.
- **genolytics.executions:** Defines and executes the desired benchmarking tests. 

As the Bayesian NMF models have been executed in R, they are not part of the `genolytics` package. However, `genolytics.bnmf_dive` collects scripts used to analyze the Bayesian results. The Bayesian NMF scripts can be found in `examples`.
### examples
For convenient try-outs, I added example scripts for each model in `examples`. 
Instead of executing the whole benchmarking procedure, single runs will be executed. The scripts are already pre-parameterized, but can be easily manipulated to test different cases.
