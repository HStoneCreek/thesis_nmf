from sqlalchemy import create_engine
import matplotlib.lines as mlines
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Create an engine to connect to the SQLite database
pd.set_option('display.max_columns', None)
engine = create_engine('sqlite:///pca_results.db')


def create_penalized_sparsity_plot(df1, name, k, xlim=1):
    df1 = df1.copy()
    df1["l1_ratio"] = df1["l1_ratio"].replace({0: "L2 (Ridge)", 1: "L1 (Lasso)", 0.5: "Elastic-Net"})
    df1.rename(columns={"l1_ratio": "Penalty", "Variable": "Metric", "sparsity": "Sparsity (\u0398)", "accuracy": "Accuracy"}, inplace=True)

    # Melt the DataFrame as you've done previously
    df_melted = df1.melt(id_vars=['Penalty', 'alpha_W'], value_vars=["Sparsity (\u0398)", 'Accuracy'], var_name='Metric', value_name='Value')

    # Set up the matplotlib figure
    # First plot for sparsity_H
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_melted[df_melted['Metric'] == "Sparsity (\u0398)"], x='alpha_W', y='Value', hue='Penalty', style='Metric')
    plt.title(f"Effect of penalized NMF on \u0398's sparsity for {name} data (latent {k})", fontsize=20)
    plt.xlabel('Penalty term lambda', fontsize=18)
    plt.ylabel('Sparsity (%)', fontsize=18)
    plt.legend(title='Legend', loc='upper right', bbox_to_anchor=(0.99, 0.99), borderaxespad=0.)

    plt.xlim(0, xlim)
    plt.ylim(0, 100)
    divide = 1/10 if xlim > 1 else 1/10
    plt.xticks([x * divide for x in range(0, xlim * 10 +1, 1)])
    plt.grid(True)
    plt.tight_layout()

    plt.xticks([x / 10 for x in range(0, xlim * 10 +1, 1)])
    plt.savefig(f"figs/penalized_nmf_sparsity_{name}_k_{k}.png", dpi=300)
    plt.show()

def create_penalized_accuracy_plot(df1, name, best_guess, k, xlim=1):
    df1 = df1.copy()
    df1["l1_ratio"] = df1["l1_ratio"].replace({0: "L2 (Ridge)", 1: "L1 (Lasso)", 0.5: "Elastic-Net"})
    df1.rename(columns={"l1_ratio": "Penalty", "Variable": "Metric", "sparsity": "Sparsity (\u0398)",
                        "accuracy": "Accuracy"}, inplace=True)

    # Melt the DataFrame as you've done previously
    df_melted = df1.melt(id_vars=['Penalty', 'alpha_W'], value_vars=["Sparsity (\u0398)", 'Accuracy'],
                         var_name='Metric', value_name='Value')
    # Second plot for accuracy
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_melted[df_melted['Metric'] == 'Accuracy'], x='alpha_W', y='Value', hue='Penalty', style='Metric')
    plt.title(f"Accuracy of penalized NMF applied on {name} data (latent {k})", fontsize=20)
    plt.xlabel('Penalty term lambda', fontsize=18)
    plt.ylabel('Accuracy (%)', fontsize=18)
    plt.axhline(y=best_guess, color='r', linestyle='--')
    handles, labels = plt.gca().get_legend_handles_labels()
    best_guess_line = mlines.Line2D([], [], color='red', linestyle='--', label='Best guess')
    handles.append(best_guess_line)
    plt.legend(handles=handles, title='Legend', loc='upper right', bbox_to_anchor=(0.99, 0.99), borderaxespad=0.)
    plt.ylim(0, 100)
    plt.yticks(range(0, 101, 10))
    plt.xlim(0, xlim)
    divide = 1/10 if xlim > 1 else 1/10
    plt.xticks([x * divide for x in range(0, xlim * 10 +1, 1)])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"figs/penalized_nmf_accuracy_{name}_k_{k}.png", dpi=300)
    plt.show()



# Define your SQL query
query = """SELECT dataset, model_name, rand_score, accuracy, alpha_W, alpha_H, l1_ratio, sparsity, sparsity_H, retained_components, purity, entropy, dropout, solver, beta_loss
        FROM benchmark_runs_nmf
        WHERE dataset = 'synthetic'"""

K = 3
df = pd.read_sql_query(query, engine.raw_connection())

df = df[(df["retained_components"] == K)& (df["solver"] == "cd") & (df["beta_loss"] == "frobenius")]

# [df["dataset"] == "leukemia"]
leukemia = df.groupby(["dataset", 'model_name', "retained_components", "l1_ratio", "alpha_W", 'alpha_H', 'solver', 'beta_loss'])[['rand_score', 'accuracy', 'sparsity', 'purity', 'entropy']].mean().reset_index()
#print(leukemia)
leukemia["accuracy"] *= 100
leukemia["sparsity"] *= 100

df1 = leukemia[leukemia["dataset"] == "synthetic"].copy()

idx = df1.groupby(['l1_ratio', 'solver', 'beta_loss'])['accuracy'].idxmax()
highest_accuracy_df = df1.loc[idx]
print(highest_accuracy_df)

create_penalized_sparsity_plot(df1=df1, name="synthetic", k=K)

create_penalized_accuracy_plot(df1=df1, name="synthetic", best_guess=30/90 * 100, k=K)


# Define your SQL query
query = """SELECT dataset, model_name, rand_score, accuracy, alpha_W, alpha_H, l1_ratio, sparsity, retained_components, purity, entropy, dropout, solver, beta_loss
        FROM benchmark_runs_nmf
        WHERE dataset = 'leukemia' AND dropout IS NOT NULL"""

df = pd.read_sql_query(query, engine.raw_connection())
K = 6
df = df[(df["retained_components"] == K) & (df["solver"] == "cd") & (df["beta_loss"] == "frobenius")]

# [df["dataset"] == "leukemia"]
leukemia = df.groupby(["dataset", 'model_name', "retained_components", "l1_ratio", "alpha_W", 'alpha_H',  'solver', 'beta_loss'])[['rand_score', 'accuracy', 'sparsity','purity', 'entropy']].mean().reset_index()
leukemia = leukemia.sort_values("l1_ratio")
print(leukemia[leukemia["model_name"] == 'NMF'])
idx = leukemia.groupby(['l1_ratio', 'solver', 'beta_loss'])['accuracy'].idxmax()
highest_accuracy_df = leukemia.loc[idx]
print(highest_accuracy_df)
leukemia["accuracy"] *= 100
leukemia["sparsity"] *= 100


df1 = leukemia[leukemia["dataset"] == "leukemia"].copy()

create_penalized_sparsity_plot(df1=df1, name="leukemia", xlim=3, k=K)

create_penalized_accuracy_plot(df1=df1, name="leukemia", best_guess=19/38 * 100, xlim=3, k=K)


query = """SELECT dataset, model_name, rand_score, accuracy, alpha_W, alpha_H, l1_ratio, sparsity, retained_components, purity, entropy, dropout, solver, beta_loss
        FROM benchmark_runs_nmf
        WHERE dataset = 'barley' AND dropout IS NOT NULL"""

df = pd.read_sql_query(query, engine.raw_connection())

# [df["dataset"] == "leukemia"]
barley = df.groupby(["dataset", 'model_name', "retained_components", "l1_ratio", "alpha_W", 'alpha_H',  'solver', 'beta_loss'])[['rand_score', 'accuracy', 'sparsity','purity', 'entropy']].mean().reset_index()

barley["accuracy"] *= 100
barley["sparsity"] *= 100


df1 = barley[barley["dataset"] == "barley"].copy()

create_penalized_sparsity_plot(df1=df1, name="barley", xlim=10, k=K)

create_penalized_accuracy_plot(df1=df1, name="barley", best_guess=120 /371 * 100, xlim=10, k=K)


