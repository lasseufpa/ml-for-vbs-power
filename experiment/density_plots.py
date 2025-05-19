"""
Script to generate density plots of performance-related variables for different CPU platforms
in uplink (UL) experiments. The dataset is filtered to include only valid experiments with
adaptive MCS (fixed_mcs_flag == 0) and 50 MHz bandwidth.

The final figure compares the distribution of the following variables across CPU platforms:
- Airtime
- MCS (Modulation and Coding Scheme)
- SNR (Signal-to-Noise Ratio)
- Power Consumption

The plot is saved to 'in_out_files/figures/density_plot.png'.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

# Set default font size for plots
font_size = 16
plt.rcParams.update({"font.size": font_size})

# Define the relevant columns to load from the CSV file
selected_columns = [
    "cpu_platform",
    "airtime",
    "mean_snr",
    "mean_used_mcs",
    "rapl_power",
    "fixed_mcs_flag",
    "failed_experiment",
    "BW",
]

# Load dataset using only the selected columns
df = pd.read_csv(
    "in_out_files/dataset_ul.csv", usecols=lambda column: column in selected_columns
)

# Rename columns to more user-friendly labels
df.rename(
    columns={
        "airtime": "Airtime",
        "mean_snr": "SNR",
        "mean_used_mcs": "MCS",
        "rapl_power": "Power Consumption",
    },
    inplace=True,
)

# Replace full CPU names with shorter platform identifiers
df["cpu_platform"] = df["cpu_platform"].replace(
    {
        "Intel(R) Core(TM) i7-8559U CPU @ 2.70GHz": "NUC1",
        "Intel(R) Core(TM) i7-8650U CPU @ 1.90GHz": "NUC2",
        "Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz": "Server1",
        "Intel(R) Core(TM) i7-9700 CPU @ 3.00GHz": "Server2",
    }
)

# Filter the dataset: only valid experiments with adaptive MCS and 50 MHz bandwidth
df_cpu = df.loc[
    (df["fixed_mcs_flag"] == 0) & (df["failed_experiment"] == 0) & (df["BW"] == 50)
]

# Variables to analyze
columns = ["Airtime", "MCS", "SNR", "Power Consumption"]
platforms = df_cpu["cpu_platform"].unique()

# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Generate a density histogram for each variable and CPU platform
for i, col in enumerate(columns):
    ax = axs[i // 2, i % 2]  # Determine subplot position

    for cpu in platforms:
        sub_df = df_cpu[df_cpu["cpu_platform"] == cpu]
        sn.histplot(sub_df[col], kde=True, ax=ax, label=cpu, stat="density")

    ax.set_xlabel(col)
    ax.set_ylabel("Density")
    ax.legend(title="CPU")

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig("in_out_files/figures/density_plot.png")
