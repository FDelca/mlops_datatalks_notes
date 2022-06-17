import os
import json
import base64
import boto3 # To send this to another data stream kinesis

import mlflow

kinesis_client = boto3.client('kinesis') # this gives us a client to interact with kinesis stream

PREDICTIONS_STREAM_NAME = os.getenv('PREDICTIONS_STREAM_NAME', 'ride_predictions')

# If the tracking server is NOT running
RUN_ID = 'ce475308510d405a8556e734d60feccd'
logged_model = f"./mlruns/1/{RUN_ID}/artifacts/model"
model = mlflow.pyfunc.load_model(logged_model)

TEST_RUN = os.getenv('TEST RUN', 'False') == 'True'


# Data Transformation
def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID']) # This approach will turn everything into a string
    features['trip_distance'] = ride['trip_distance']
    return features

# Create a function to predict
def predict(features):
    pred = model.predict(features)
    return float(pred[0])


def lambda_handler(event, context):

    # print(json.dumps(event))
    
    predictions_events = []
    
    
    for record in event['Records']:
        encoded_data = record['kinesis']['data']
        decoded_data = base64.b64decode(encoded_data).decode('utf-8')
        ride_event = json.loads(decoded_data)
        
        # print(ride_event)
        ride = ride_event['ride']
        ride_id = ride_event['ride_id']
        
        features = prepare_features(ride)
        prediction = predict(features)
        
        prediction_event = {
            'model':'ride_duration_prediction_model',
            'version':'v1',
            'prediction':{
            'ride_duration': prediction,
            'ride_id': ride_id},
        }

        # To connect with other kinesis stream - to each prediction event we will put the result to another stream
        if not TEST_RUN:
            kinesis_client.put_record(
                    StreamName=PREDICTIONS_STREAM_NAME,
                    Data=json.dumps(prediction_event),
                    PartitionKey=str(ride_id)
                    )
    
        predictions_events.append(prediction_event)
        

        

    return {
        'predictions': predictions_events
    }