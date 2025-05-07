import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from scipy.stats import uniform, randint
from sklearn.preprocessing import MinMaxScaler
import warnings
from datetime import datetime
from pytz import timezone
import sys

warnings.filterwarnings("ignore")

def current_time():
    now = datetime.now(timezone('America/Belem'))
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

def evaluate_models_by_platform(features, target, best_metric='mse', n_iter=10):
    metrics = {
        'mae': mean_absolute_error,
        'rmse': root_mean_squared_error,
        'mape': mean_absolute_percentage_error
    }

    platforms = df['cpu_platform'].unique()

    for platform in platforms:
        print(f"{'='*50}\nProcessing platform: {platform}\n{'='*50}")

        df_platform = df.loc[
            (df['fixed_mcs_flag'] == 0) &
            (df['failed_experiment'] == 0) &
            (df['BW'] == 50) &
            (df['cpu_platform'] == platform)
        ].copy()

        if df_platform.empty:
            print(f"No data available for platform: {platform}")
            continue

        scaler = MinMaxScaler()
        df_platform[features] = scaler.fit_transform(df_platform[features])

        X = np.array(df_platform[features])
        y = np.array(df_platform[target])

        n_samples = len(X)
        n_test = int(np.floor(n_samples * 0.2))
        n_train = int(np.floor(n_samples * 0.8))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n_test, train_size=n_train, random_state=42)

        model_results = {}

        for model_name, model in models.items():
            try:
                print(f"{current_time()} - Starting training of {model_name}...")

                model_params = param_distribs.get(model_name, {})

                search = RandomizedSearchCV(
                    model, model_params, n_iter=n_iter, random_state=42,
                    scoring='neg_mean_squared_error', n_jobs=-1
                )
                search.fit(X_train, y_train)
                y_pred = search.best_estimator_.predict(X_test)

                score = metrics[best_metric](y_test, y_pred)

                model_results[model_name] = {'metric': score, 'params': search.best_params_}
                print(f"{current_time()} - {model_name} training finished.")

            except Exception as e:
                print(f"Error training the model {model_name}: {e}")

        if model_results:
            best_model_name = min(model_results, key=lambda x: model_results[x]['metric'])
            best_score = model_results[best_model_name]['metric']
            best_params = model_results[best_model_name]['params']

            print(f"\nBest model for {platform}: {best_model_name} - {best_metric.upper()}: {best_score:.4f}")

            print("\nAll model results:")
            for model_name, result in model_results.items():
                print(f"{model_name} - {best_metric.upper()}: {result['metric']:.4f} - Parameters: {format_params(model_name, models, result['params'])}")
        else:
            print(f"No models were successfully trained for {platform}.")

with open('in_out_files/random_search_output.txt', "w") as f:
    sys.stdout = f

    features = ['airtime', 'mean_used_mcs', 'mean_snr']
    target = 'rapl_power'
    config_cols = ['cpu_platform', 'fixed_mcs_flag', 'failed_experiment', 'BW']
    df = pd.read_csv('in_out_files/dataset_ul.csv', usecols=features + config_cols + [target])

    df['cpu_platform'] = df['cpu_platform'].replace({
        'Intel(R) Core(TM) i7-8559U CPU @ 2.70GHz': 'NUC1',
        'Intel(R) Core(TM) i7-8650U CPU @ 1.90GHz': 'NUC2',
        'Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz': 'Server1',
        'Intel(R) Core(TM) i7-9700 CPU @ 3.00GHz': 'Server2'
    })

    models = {
        'xgb.XGBRegressor': xgb.XGBRegressor(random_state=42),
        'LinearRegression': LinearRegression(),
        'MLPRegressor': MLPRegressor(max_iter=500, early_stopping=True, validation_fraction=0.1, random_state=42)
    }

    param_distribs = {
        'xgb.XGBRegressor': {
            'n_estimators': randint(50, 200),
            'learning_rate': uniform(0.01, 0.2),
            'reg_alpha': uniform(0, 1),
            'reg_lambda': uniform(0, 1),
            'scale_pos_weight': [1, 2],
            'booster': ['gbtree', 'gblinear', 'dart'],
        },
        'LinearRegression': {
            'fit_intercept': [True, False],
        },
        'MLPRegressor': {
            'hidden_layer_sizes': [(i,) for i in range(1, 200, 2)],
            'activation': ['relu', 'tanh', 'logistic'],
            'solver': ['adam', 'lbfgs', 'sgd'],
            'alpha': uniform(0.0001, 0.1),
            'learning_rate_init': [0.001, 0.01, 0.1],
            'beta_1': uniform(0.8, 0.199),
            'beta_2': uniform(0.9, 0.0999),
        }
    }

    evaluate_models_by_platform(features, target, best_metric='mape', n_iter=100000)

sys.stdout = sys.__stdout__
print("Execution finished.")