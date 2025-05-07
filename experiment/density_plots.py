import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

font_size = 16
plt.rcParams.update({"font.size": font_size})

cols = [
    "cpu_platform",
    "airtime",
    "mean_snr",
    "mean_used_mcs",
    "rapl_power",
    "fixed_mcs_flag",
    "failed_experiment",
    "BW",
]

df = pd.read_csv("in_out_files/dataset_ul.csv", usecols=cols)

df.rename(
    columns={
        "airtime": "Airtime",
        "mean_snr": "SNR",
        "mean_used_mcs": "MCS",
        "rapl_power": "Power Consumption",
    },
    inplace=True,
)

df["cpu_platform"] = df["cpu_platform"].replace(
    {
        "Intel(R) Core(TM) i7-8559U CPU @ 2.70GHz": "NUC1",
        "Intel(R) Core(TM) i7-8650U CPU @ 1.90GHz": "NUC2",
        "Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz": "Server1",
        "Intel(R) Core(TM) i7-9700 CPU @ 3.00GHz": "Server2",
    }
)

df_cpu = df.loc[
    (df["fixed_mcs_flag"] == 0) & (df["failed_experiment"] == 0) & (df["BW"] == 50)
]

columns = ["Airtime", "MCS", "SNR", "Power Consumption"]
platforms = df_cpu["cpu_platform"].unique()

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

for i, col in enumerate(columns):
    ax = axs[i // 2, i % 2]

    for cpu in platforms:
        sub_df = df_cpu[df_cpu["cpu_platform"] == cpu]
        sn.histplot(sub_df[col], kde=True, ax=ax, label=cpu, stat="density")

    ax.set_xlabel(col)
    ax.set_ylabel("Density")
    ax.legend(title="CPU")

plt.tight_layout()
plt.savefig("in_out_files/figures/density_plot.png")
