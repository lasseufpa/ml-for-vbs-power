"""
This script performs model training and evaluation for power consumption prediction
across multiple CPU platforms. Three regression models (Linear Regression, XGBoost,
and Neural Network) are pre-configured with optimized hyperparameters for each platform.

The script:
- Loads and filters the dataset
- Splits data into train/test sets
- Applies MinMax scaling
- Fits each model using the specified hyperparameters
- Evaluates using MAE, RMSE, and MAPE
- Saves predictions and metrics
- Generates scatter plots comparing predicted vs. actual power consumption per platform
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from numpy import nan
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             root_mean_squared_error)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")

# Feature and target definitions
features = ["airtime", "mean_snr", "mean_used_mcs"]
target = "rapl_power"
config_cols = ["cpu_platform", "fixed_mcs_flag", "failed_experiment", "BW"]

cols = features + config_cols + [target]

# Load only necessary columns from CSV
df = pd.read_csv("in_out_files/dataset_ul.csv", usecols=lambda column: column in cols)

# Simplify CPU platform names
df["cpu_platform"] = df["cpu_platform"].replace(
    {
        "Intel(R) Core(TM) i7-8559U CPU @ 2.70GHz": "NUC1",
        "Intel(R) Core(TM) i7-8650U CPU @ 1.90GHz": "NUC2",
        "Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz": "Server1",
        "Intel(R) Core(TM) i7-9700 CPU @ 3.00GHz": "Server2",
    }
)

# Predefined hyperparameters for each model and platform (from prior optimization)
hyperparams = {
    "NUC1": {
        "LR": LinearRegression(fit_intercept=True),
        "XGB": xgb.XGBRegressor(
            colsample_bytree=1.0,
            gamma=0,
            learning_rate=0.1,
            max_depth=9,
            min_child_weight=6,
            n_estimators=151,
            reg_alpha=0,
            reg_lambda=10,
            subsample=0.8,
            missing=nan,
            random_state=42,
        ),
        "NN": MLPRegressor(
            solver="adam",
            learning_rate_init=0.01,
            hidden_layer_sizes=(90, 100, 60),
            beta_2=0.99,
            beta_1=0.9,
            alpha=0.01,
            activation="relu",
            early_stopping=True,
            max_iter=500,
            random_state=42,
        ),
    },
    "NUC2": {
        "LR": LinearRegression(fit_intercept=True),
        "XGB": xgb.XGBRegressor(
            colsample_bytree=0.7,
            gamma=0,
            learning_rate=0.3,
            max_depth=3,
            min_child_weight=6,
            n_estimators=144,
            reg_alpha=0.01,
            reg_lambda=0.1,
            subsample=0.7,
            missing=nan,
            random_state=42,
        ),
        "NN": MLPRegressor(
            solver="adam",
            learning_rate_init=0.01,
            hidden_layer_sizes=(60, 30, 95),
            beta_2=0.999,
            beta_1=0.9,
            alpha=0.0001,
            activation="relu",
            early_stopping=True,
            max_iter=500,
            random_state=42,
        ),
    },
    "Server1": {
        "LR": LinearRegression(fit_intercept=True),
        "XGB": xgb.XGBRegressor(
            colsample_bytree=1.0,
            gamma=0,
            learning_rate=0.3,
            max_depth=4,
            min_child_weight=1,
            n_estimators=192,
            reg_alpha=0.01,
            reg_lambda=0.1,
            subsample=0.8,
            missing=nan,
            random_state=42,
        ),
        "NN": MLPRegressor(
            solver="adam",
            learning_rate_init=0.001,
            hidden_layer_sizes=(40, 70, 65),
            beta_2=0.99,
            beta_1=0.9,
            alpha=0.001,
            activation="relu",
            early_stopping=True,
            max_iter=500,
            random_state=42,
        ),
    },
    "Server2": {
        "LR": LinearRegression(fit_intercept=True),
        "XGB": xgb.XGBRegressor(
            colsample_bytree=1.0,
            gamma=0,
            learning_rate=0.05,
            max_depth=3,
            min_child_weight=1,
            n_estimators=193,
            reg_alpha=0,
            reg_lambda=0.1,
            subsample=0.7,
            missing=nan,
            random_state=42,
        ),
        "NN": MLPRegressor(
            solver="sgd",
            learning_rate_init=0.001,
            hidden_layer_sizes=(15, 30, 85),
            beta_2=0.999,
            beta_1=0.95,
            alpha=0.01,
            activation="relu",
            early_stopping=True,
            max_iter=500,
            random_state=42,
        ),
    },
}

# List of platforms found in the dataset
platforms = df["cpu_platform"].unique()
final_results = []

# Loop through each CPU platform and evaluate models
for cpu in platforms:
    df_cpu = df.copy()
    df_cpu = df_cpu.loc[
        (df_cpu["fixed_mcs_flag"] == 0)
        & (df_cpu["failed_experiment"] == 0)
        & (df_cpu["BW"] == 50)
        & (df_cpu["cpu_platform"] == cpu)
    ]

    if not df_cpu.empty:
        # Prepare features and labels
        X = np.array(df_cpu[features])
        y = np.array(df_cpu[target])

        # Split data into training and testing sets
        n_samples = len(X)
        n_test = int(np.floor(n_samples * 0.2))
        n_train = int(np.floor(n_samples * 0.8))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=n_test, train_size=n_train, random_state=42
        )

        # Normalize features
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Evaluate all models for this platform
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

# Convert results to DataFrame and save
results_df = pd.DataFrame(final_results)
results_df.to_csv("in_out_files/train_test_output.csv", index=False)

# Plotting: scatter plots of predicted vs actual power per CPU platform
font_size = 11
plt.rcParams.update({"font.size": font_size})

for cpu in platforms:
    plt.figure(figsize=(6, 4))

    all_preds = []
    all_tests = []

    # Plot predictions from each model
    for index, row in results_df[results_df["CPU"] == cpu].iterrows():
        model = row["Model"]
        y_test = row["y_test"]
        y_pred = row["y_pred"]
        plt.scatter(y_pred, y_test, label=model, alpha=0.6)
        all_preds.extend(y_pred)
        all_tests.extend(y_test)

    # Add identity line for reference
    min_val = min(min(all_tests), min(all_preds))
    max_val = max(max(all_tests), max(all_preds))
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        color="orange",
        linestyle="--",
        label="Identity Line",
    )

    # Final plot adjustments
    plt.xlabel("Predicted Power [W]", fontsize=font_size)
    plt.ylabel("Measured Power [W]", fontsize=font_size)
    plt.legend(fontsize=font_size)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"in_out_files/figures/scatter_plot-{cpu}.png")
