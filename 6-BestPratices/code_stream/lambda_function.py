import os
import json
import base64
import boto3
import mlflow

# To interact with Kinesis Stream
kinesis_client = boto3.client('kinesis')
PREDICTIONS_STREAM_NAME = os.getenv('PREDICTIONS_STREAM_NAME', 'ride-predictions')

# Load the model from s3 bucket
RUN_ID = os.getenv('RUN_ID', '8b4afe073de2423cad4b858170ac574f')
logged_model = f"s3://mlflow-artifacts-rmt/0/{RUN_ID}/artifacts/model"

model = mlflow.pyfunc.load_model(logged_model)

TEST_RUN = os.getenv('TEST_RUN', 'False') == 'True'

def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features

def predict(features):
    pred = model.predict(features)
    return float(pred[0])

def lambda_handler(event, context):

    # print(json.dumps(event))
    
    # We have multiple predictions - because lambda might consume 100 events
    predictions_events = []
    for record in event['Records']:
        encoded_data = record['kinesis']['data']
        decoded_data = base64.b64decode(encoded_data).decode('utf-8')
        ride_event = json.loads(decoded_data) # returns a json file
        # print(ride_event)
    
    
        ride = ride_event['ride']
        ride_id = ride_event['ride_id']

        features = prepare_features(ride)
        prediction = predict(features)
        
        # We will send this to a different stream - to show to the client
        prediction_event = {
            'model': 'ride_duration_prediction_model',
            'version': '123',
            'prediction': {
                'ride_duration': prediction,
                'ride_id': ride_id
            },
        }
        
        if not TEST_RUN:
            # After each prediction event we will send data to Kinesis Data Stream
            response = kinesis_client.put_record(
                StreamName=PREDICTIONS_STREAM_NAME,
                Data=json.dumps(prediction_event),
                PartitionKey=str(ride_id),
            )
        
        predictions_events.append(prediction_event)
        

    return {
        'predictions': predictions_events,
    }