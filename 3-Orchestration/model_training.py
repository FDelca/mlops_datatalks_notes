from audioop import add
import pandas as pd
import pickle
import os

from sklearn.feature_extraction import DictVectorizer
# ^from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error

import xgboost as xgb

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

# Set environment variables
mlflow.set_tracking_uri("sqlite:///backend.db")
mlflow.set_experiment("nyc-taxi-experiment")

def read_dataframe(filename):
    df = pd.read_parquet(filename)

    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df

def add_features(train_path="./data/green_tripdata_2021-01.parquet",
                 val_path="./data/green_tripdata_2021-02.parquet"):
    df_train = read_dataframe(train_path)
    df_val = read_dataframe(val_path)

    print(len(df_train))
    print(len(df_val))

    df_train['PU_DO'] = df_train['PULocationID'] + '_' + df_train['DOLocationID']
    df_val['PU_DO'] = df_val['PULocationID'] + '_' + df_val['DOLocationID']

    categorical = ['PU_DO'] #'PULocationID', 'DOLocationID']
    numerical = ['trip_distance']

    dv = DictVectorizer()

    train_dicts = df_train[categorical + numerical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[categorical + numerical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)

    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    return X_train, X_val, y_train, y_val, dv

def train_model_search(train, valid, y_val):
    def objective(params):
        with mlflow.start_run():
            mlflow.set_tag("model", "xgboost")
            mlflow.log_params(params)
            booster = xgb.train(
                params=params,
                dtrain=train,
                num_boost_round=100,
                evals=[(valid, 'validation')],
                early_stopping_rounds=50
            )
            y_pred = booster.predict(valid)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mlflow.log_metric("rmse", rmse)

        return {'loss': rmse, 'status': STATUS_OK}

    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
        'learning_rate': hp.loguniform('learning_rate', -3, 0),
        'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
        'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
        'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
        'objective': 'reg:linear',
        'seed': 42
    }

    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=10,
        trials=Trials()
    )
    return

def train_best_model(train, valid, y_val, dv):
    with mlflow.start_run():
        # Other way and better way of doing this is to use MLClient
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        ### Select the run with the lowest rmse
        client = MlflowClient()
        experiment = client.get_experiment_by_name('nyc-taxi-experiment')
        runs = client.search_runs(
            experiment_ids=experiment.experiment_id,
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=1,
            order_by=["metrics.rmse ASC"]
            )

        best_params = runs[0].data.params
        # Change params type to enter in the model
        for key in best_params:
            if key == 'seed' or key == 'max_depth':
                best_params[key] = int(best_params[key])
            
            elif key != 'objective':
                best_params[key] = float(best_params[key])
        ###

        mlflow.log_params(best_params)
        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=100,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

if __name__ == "__main__":

    # Requirements
    folder = 'models'
    if not os.path.exists(folder):
        print(f'{folder} created')
        os.mkdir(folder)

    # Create Features
    X_train, X_val, y_train, y_val, dv = add_features()
    
    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)

    # Hyperparameter optimization - using hyperopt
    train_model_search(train, valid, y_val)
    
    # The hyperparameters are defined - try to use MLClient
    train_best_model(train, valid, y_val, dv)