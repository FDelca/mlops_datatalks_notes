import os
import argparse
import pandas as pd
import pickle
import datetime
from dateutil.relativedelta import relativedelta


from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner


@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):

    logger = get_run_logger()
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):
    logger = get_run_logger()

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return

@task
def get_paths(date):
    logger = get_run_logger()

    if date == None:
        date = datetime.date.today()
    else:
        date = pd.to_datetime(date)

    date_train = date - relativedelta(months=2)
    date_val = date - relativedelta(months=1)

    date_train = date_train.strftime("%Y-%m")
    date_val = date_val.strftime("%Y-%m")
    
    # Complete val and train path
    train_path = 'data/fhv_tripdata_'+date_train+'.parquet'
    val_path = 'data/fhv_tripdata_'+date_val+'.parquet'  
    
    if not os.path.exists(train_path):
        url = f"https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{date_train}.parquet"
        try:
            os.system(f"wget {url} -P \data")
        except:
            logger.warning(f"{url} not available")
            raise Exception('Train data not available')
    
    if not os.path.exists(val_path):
        url = f"https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{date_val}.parquet"
        try:
            os.system(f"wget {url} -P \data")
        except:
            logger.warning(f"{url} not available")
            raise Exception('Val data not available')


        
    logger.info(f"Train path: {train_path}")
    logger.info(f"Val path: {val_path}")

    return train_path, val_path


@flow(task_runner=SequentialTaskRunner())
def main(date="2021-08-15"):

    train_path, val_path = get_paths(date).result()

    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)

    # Save models
    with open(f"models/model-{date}.bin", 'wb') as f_out:
        pickle.dump(lr, f_out)

    with open(f"models/dv-{date}.bin", 'wb') as f_out:
        pickle.dump(dv, f_out)


from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner

DeploymentSpec(
    flow=main,
    name='model_training',
    schedule=CronSchedule(cron='0 9 15 * *'),
    flow_runner=SubprocessFlowRunner(),
    tags=['first_model'],
)


# For Question 1/2/3
#if __name__ == '__main__':
#
#   parser = argparse.ArgumentParser()
#    parser.add_argument(
#        "--date",
#        default=None,
#        help="Date to catch the data to train and validate"
#    )
#    args = parser.parse_args()
#
#    main(args.date)
