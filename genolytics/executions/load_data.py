import os

import pandas as pd
import numpy as np

def find_project_root(start_path, project_root_name='thesis_nmf'):
    current_path = start_path
    while current_path != os.path.dirname(current_path):  # While not at the root directory
        if os.path.basename(current_path) == project_root_name:
            return current_path
        current_path = os.path.dirname(current_path)
    raise FileNotFoundError(f"Could not find the {project_root_name} directory")

def compute_kinship_matrix(genotypes):
    # Convert to float for numerical operations
    genotypes = genotypes.astype(float)

    # Calculate the allele frequencies (p)
    # Mean along the rows (individuals), divide by 2 since the counts are 0, 1, 2
    p = genotypes.mean(axis=0) / 2

    # Center the genotypes matrix by subtracting 2*p from each element
    X_centered = genotypes - 2 * p

    # Calculate the variance for each SNP
    variances = 2 * p * (1 - p)
    variances[variances == 0] = 1  # To avoid division by zero

    # Scale the centered matrix
    X_scaled = X_centered / np.sqrt(variances)

    # Compute the kinship matrix using matrix multiplication
    # Normalize by the number of SNPs
    kinship_matrix = np.dot(X_scaled, X_scaled.T) / genotypes.shape[1]

    return kinship_matrix

def load_leukemia_data():

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Find the thesis_nmf directory
    thesis_nmf_dir = find_project_root(current_dir)
    X_file = os.path.join(thesis_nmf_dir, 'leukemia', 'ALL_AML_data.txt')
    y_file = os.path.join(thesis_nmf_dir, 'leukemia', 'ALL_AML_samples.txt')
    X = pd.read_csv(X_file, sep="\t", header=None)
    y = pd.read_csv(y_file, sep="\t", names=["Sample"])
    y = y.dropna()
    y["Sample"] = y["Sample"].astype(str)

    y['Cell_Type'] = y['Sample'].apply(lambda x: 'AML' if x.startswith('AML') else (
        'B-cell' if 'B-cell' in x else ('T-cell' if 'T-cell' in x else 'none')))
    y["cluster"] = pd.factorize(y['Cell_Type'])[0]
    X = X.transpose().values
    return X, y["cluster"].values


def load_barley(drop_meta: bool = True,
                ):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Find the thesis_nmf directory
    thesis_nmf_dir = find_project_root(current_dir)
    X_file = os.path.join(thesis_nmf_dir, 'barley', 'whealbi_SNP.csv')
    y_file = os.path.join(thesis_nmf_dir, 'barley', 'barley_meta_data.xlsx')

    data = pd.read_csv(X_file, sep=",")
    data = data.rename(columns={"Unnamed: 0": "identifier"})

    meta = pd.read_excel(y_file, skiprows=1)
    meta = meta.rename(columns={"WHEALBI Accession number (full set of 512 lines)": "identifier",
                                "Sub-population assignment based on exome data analysis (for set of 371 domesticated barleys)":
                                    "subpopulation"})
    cols = ['Row type (for set of 371 domesticated barleys)',
            'Breeding history (where known, for set of 371 domesticated barleys)',
            'Growth habit (where known, for set of 371 domesticated barleys)',
            'Country of origin (where available, passport data)']
    data = pd.merge(data, meta[['identifier', 'subpopulation'] + cols ], on="identifier", how="inner")

    data = data.rename(columns={
        'Row type (for set of 371 domesticated barleys)': 'Row_type',
        'Breeding history (where known, for set of 371 domesticated barleys)': 'Breeding History',
        'Growth habit (where known, for set of 371 domesticated barleys)': 'Growth habit',
        'Country of origin (where available, passport data)': 'Country'
    })
    data = data.sort_values("subpopulation").reset_index(drop=True)
    data["Sample"] = pd.factorize(data['subpopulation'])[0]

    if drop_meta:
        df = data.drop(["identifier", "subpopulation", 'Sample', 'Row_type', 'Breeding History', 'Growth habit', 'Country'], axis=1).copy()
    else:
        df = data.copy()

    return df, data['Sample']