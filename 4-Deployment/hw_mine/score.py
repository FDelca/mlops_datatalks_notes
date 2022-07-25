import pickle
import pandas as pd
import sys

def read_data(filename, categorical):
    df = pd.read_parquet(filename)

    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def load_model():
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)
    return dv, lr

def apply_model(df, dv, lr, categorical):
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)
    return y_pred

def save_output(df, y_pred, output_file):

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred
    #df_result.to_parquet(
    #    output_file,
    #    engine='pyarrow',
    #    compression=None,
    #    index=False
    #)


def run():
    
    year = int(sys.argv[1]) # 2021
    month = int(sys.argv[2]) # 2

    taxi_type = 'fhv'
    categorical = ['PUlocationID', 'DOlocationID']

    input_file = f'https://s3.amazonaws.com/nyc-tlc/trip+data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    print(f"reading data {taxi_type}_tripdata_{year:04d}-{month:02d}.parquet")
    df = read_data(input_file, categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    print(f"loading model ...")
    dv, lr = load_model()
    
    print(f"apply model ...")
    y_pred = apply_model(df, dv, lr, categorical)
    print(f'Mean result: {y_pred.mean()}')

    print('saving output results')
    output_file = f"output/fhv/{year:04d}_{month:02d}.parquet"
    save_output(df, y_pred, output_file)

if __name__ == '__main__':
    run()
