import sys
import warnings
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import xgboost as xgb
from pytz import timezone
from scipy.stats import randint, uniform
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def current_time():
    now = datetime.now(timezone("America/Belem"))
    return str(now.strftime("%Y-%m-%d_%H-%M-%S"))


def format_params(model_name, models, params):
    param_str = ", ".join(f"{key}={repr(value)}" for key, value in params.items())

    model = models[model_name]
    defaults = type(model)().get_params()
    custom = model.get_params()
    custom_param = ""
    for param, val in custom.items():
        if param not in defaults or defaults[param] != val:
            custom_param += f", {param}={val}"

    return f"{model_name}({param_str}{custom_param})"


def evaluate_models_by_platform(features, target, best_metric="mse"):
    metrics = {
        "mae": mean_absolute_error,
        "rmse": root_mean_squared_error,
        "mape": mean_absolute_percentage_error,
    }

    if best_metric == "mape":
        scoring = make_scorer(mean_absolute_percentage_error, greater_is_better=False)
    elif best_metric == "mae":
        scoring = make_scorer(mean_absolute_error, greater_is_better=False)
    elif best_metric == "rmse":
        scoring = make_scorer(root_mean_squared_error, greater_is_better=False)
    else:
        raise ValueError(f"Unsupported metric: {best_metric}")

    platforms = df["cpu_platform"].unique()

    for platform in platforms:
        print(f"{'='*50}\nProcessing platform: {platform}\n{'='*50}")

        df_platform = df.loc[
            (df["fixed_mcs_flag"] == 0)
            & (df["failed_experiment"] == 0)
            & (df["BW"] == 50)
            & (df["cpu_platform"] == platform)
        ].copy()

        if df_platform.empty:
            print(f"No data available for platform: {platform}")
            continue

        X = np.array(df_platform[features])
        y = np.array(df_platform[target])

        n_samples = len(X)
        n_test = int(np.floor(n_samples * 0.2))
        n_train = int(np.floor(n_samples * 0.8))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=n_test, train_size=n_train, random_state=42
        )

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model_results = {}

        for model_name, model in models.items():
            try:
                print(f"{current_time()} - Starting training of {model_name}...")

                model_params = param_distribs.get(model_name, {})

                if model_name == "LinearRegression":
                    search = GridSearchCV(
                        model,
                        model_params,
                        scoring=scoring,
                        n_jobs=-1,
                    )
                else:
                    search = RandomizedSearchCV(
                        model,
                        model_params,
                        n_iter=n_iter_map[model_name],
                        random_state=42,
                        scoring=scoring,
                        n_jobs=-1,
                    )

                search.fit(X_train, y_train)
                y_pred = search.best_estimator_.predict(X_test)

                score = metrics[best_metric](y_test, y_pred)

                model_results[model_name] = {
                    "metric": score,
                    "params": search.best_params_,
                }
                print(f"{current_time()} - {model_name} training finished.")

            except Exception as e:
                print(f"Error training the model {model_name}: {e}")

        if model_results:
            best_model_name = min(
                model_results, key=lambda x: model_results[x]["metric"]
            )
            best_score = model_results[best_model_name]["metric"]

            print(
                f"\nBest model for {platform}: {best_model_name} - {best_metric.upper()}: {best_score:.4f}"
            )

            print("\nAll model results:")
            for model_name, result in model_results.items():
                print(
                    f"{model_name} - {best_metric.upper()}: {result['metric']:.4f} - Parameters: {format_params(model_name, models, result['params'])}"
                )
        else:
            print(f"No models were successfully trained for {platform}.")


with open("in_out_files/model_selection_output.txt", "w") as f:
    sys.stdout = f

    try:
        features = ["airtime", "mean_snr", "mean_used_mcs"]
        target = "rapl_power"
        config_cols = ["cpu_platform", "fixed_mcs_flag", "failed_experiment", "BW"]

        cols = features + config_cols + [target]
        df = pd.read_csv(
            "in_out_files/dataset_ul.csv",
            usecols=lambda column: column in cols,
        )

        df["cpu_platform"] = df["cpu_platform"].replace(
            {
                "Intel(R) Core(TM) i7-8559U CPU @ 2.70GHz": "NUC1",
                "Intel(R) Core(TM) i7-8650U CPU @ 1.90GHz": "NUC2",
                "Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz": "Server1",
                "Intel(R) Core(TM) i7-9700 CPU @ 3.00GHz": "Server2",
            }
        )

        models = {
            "xgb.XGBRegressor": xgb.XGBRegressor(random_state=42),
            "LinearRegression": LinearRegression(),
            "MLPRegressor": MLPRegressor(
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42,
            ),
        }

        hidden_layer_sizes = []
        for n_layers in range(1, 4):
            for layer_sizes in product(range(5, 101, 5), repeat=n_layers):
                hidden_layer_sizes.append(layer_sizes)

        param_distribs = {
            "xgb.XGBRegressor": {
                "n_estimators": randint(50, 201),
                "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
                "max_depth": randint(3, 11),
                "min_child_weight": randint(1, 7),
                "gamma": [0, 0.1, 0.3, 1],
                "subsample": [0.7, 0.8, 1.0],
                "colsample_bytree": [0.7, 0.8, 1.0],
                "reg_alpha": [0, 0.01, 0.1],
                "reg_lambda": [0.1, 1, 10],
            },
            "LinearRegression": {
                "fit_intercept": [True, False],
            },
            "MLPRegressor": {
                "hidden_layer_sizes": hidden_layer_sizes,
                "activation": ["relu", "tanh", "logistic"],
                "solver": ["adam", "sgd"],
                "learning_rate_init": [0.0005, 0.001, 0.01],
                "alpha": [0.0001, 0.001, 0.01],
                "beta_1": [0.9, 0.95],
                "beta_2": [0.99, 0.999],
            },
        }

        n_iter_map = {
            "xgb.XGBRegressor": 117418,  # Total: 11,741,760
            "MLPRegressor": 18187,  # Total: 1,818,720
        }

        evaluate_models_by_platform(features, target, best_metric="mape")
    finally:
        sys.stdout = sys.__stdout__

sys.stdout = sys.__stdout__
print("Execution finished.")
