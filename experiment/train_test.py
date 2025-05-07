import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from pytz import timezone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             root_mean_squared_error)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

features = ["airtime", "mean_snr", "mean_used_mcs"]
target = "rapl_power"
config_cols = ["cpu_platform", "fixed_mcs_flag", "failed_experiment", "BW"]
df = pd.read_csv(
    "in_out_files/dataset_ul.csv", usecols=features + config_cols + [target]
)

df["cpu_platform"] = df["cpu_platform"].replace(
    {
        "Intel(R) Core(TM) i7-8559U CPU @ 2.70GHz": "NUC1",
        "Intel(R) Core(TM) i7-8650U CPU @ 1.90GHz": "NUC2",
        "Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz": "Server1",
        "Intel(R) Core(TM) i7-9700 CPU @ 3.00GHz": "Server2",
    }
)

hyperparams = {
    "NUC1": {
        "LR": LinearRegression(fit_intercept=True),
        "XGB": xgb.XGBRegressor(
            random_state=42,
            booster="dart",
            learning_rate=np.float64(0.19366649440872624),
            n_estimators=114,
            reg_alpha=np.float64(0.03445612872982562),
            reg_lambda=np.float64(0.4387505218486637),
            scale_pos_weight=2,
        ),
        "NN": MLPRegressor(
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            activation="relu",
            alpha=np.float64(0.025556706015992626),
            beta_1=np.float64(0.8717195393485015),
            beta_2=np.float64(0.9596773079191824),
            hidden_layer_sizes=(113,),
            learning_rate_init=0.1,
            solver="lbfgs",
        ),
    },
    "NUC2": {
        "LR": LinearRegression(fit_intercept=True),
        "XGB": xgb.XGBRegressor(
            random_state=42,
            booster="gbtree",
            learning_rate=np.float64(0.1844533971618749),
            n_estimators=78,
            reg_alpha=np.float64(0.5304170827353661),
            reg_lambda=np.float64(0.7882212441472002),
            scale_pos_weight=1,
        ),
        "NN": MLPRegressor(
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            activation="relu",
            alpha=np.float64(0.07286827507351072),
            beta_1=np.float64(0.8383476262914942),
            beta_2=np.float64(0.9753284351908575),
            hidden_layer_sizes=(55,),
            learning_rate_init=0.001,
            solver="lbfgs",
        ),
    },
    "Server1": {
        "LR": LinearRegression(fit_intercept=True),
        "XGB": xgb.XGBRegressor(
            random_state=42,
            booster="gbtree",
            learning_rate=np.float64(0.16780581384765997),
            n_estimators=152,
            reg_alpha=np.float64(0.7588411898224708),
            reg_lambda=np.float64(0.0865104215069239),
            scale_pos_weight=2,
        ),
        "NN": MLPRegressor(
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            activation="relu",
            alpha=np.float64(0.08044028100378249),
            beta_1=np.float64(0.9391484519043694),
            beta_2=np.float64(0.9223299149135724),
            hidden_layer_sizes=(137,),
            learning_rate_init=0.1,
            solver="lbfgs",
        ),
    },
    "Server2": {
        "LR": LinearRegression(fit_intercept=True),
        "XGB": xgb.XGBRegressor(
            random_state=42,
            booster="dart",
            learning_rate=np.float64(0.10544851212228039),
            n_estimators=172,
            reg_alpha=np.float64(0.9905472744586734),
            reg_lambda=np.float64(0.9613097675711414),
            scale_pos_weight=2,
        ),
        "NN": MLPRegressor(
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            activation="relu",
            alpha=np.float64(0.05301612276992186),
            beta_1=np.float64(0.9029264376294966),
            beta_2=np.float64(0.9432064447280365),
            hidden_layer_sizes=(17,),
            learning_rate_init=0.001,
            solver="lbfgs",
        ),
    },
}

platforms = df["cpu_platform"].unique()
final_results = []

for cpu in platforms:
    df_cpu = df.copy()
    df_cpu = df_cpu.loc[
        (df_cpu["fixed_mcs_flag"] == 0)
        & (df_cpu["failed_experiment"] == 0)
        & (df_cpu["BW"] == 50)
        & (df_cpu["cpu_platform"] == cpu)
    ]

    if not df_cpu.empty:
        scaler = MinMaxScaler()
        df_cpu[features] = scaler.fit_transform(df_cpu[features])
        X = np.array(df_cpu[features])
        y = np.array(df_cpu[target])

        n_samples = len(X)
        n_test = int(np.floor(n_samples * 0.2))
        n_train = int(np.floor(n_samples * 0.8))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=n_test, train_size=n_train, random_state=42
        )

        for model_name, model in hyperparams[cpu].items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = root_mean_squared_error(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)

            final_results.append(
                {
                    "CPU": cpu,
                    "Model": model_name,
                    "y_test": y_test,
                    "y_pred": y_pred,
                    "MAE": mae,
                    "RMSE": rmse,
                    "MAPE": mape,
                }
            )

results_df = pd.DataFrame(final_results)
results_df.to_csv("in_out_files/train_test_output.csv", index=False)

font_size = 11
plt.rcParams.update({"font.size": font_size})

for cpu in platforms:
    plt.figure(figsize=(6, 4))

    all_preds = []
    all_tests = []

    for index, row in results_df[results_df["CPU"] == cpu].iterrows():
        model = row["Model"]
        y_test = row["y_test"]
        y_pred = row["y_pred"]
        plt.scatter(y_pred, y_test, label=model, alpha=0.6)
        all_preds.extend(y_pred)
        all_tests.extend(y_test)

    min_val = min(min(all_tests), min(all_preds))
    max_val = max(max(all_tests), max(all_preds))
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        color="orange",
        linestyle="--",
        label="Identity Line",
    )

    plt.xlabel("Predicted Power [W]", fontsize=font_size)
    plt.ylabel("Measured Power [W]", fontsize=font_size)
    plt.legend(fontsize=font_size)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"in_out_files/figures/scatter_plot-{cpu}.png")
