import os
import json
import boto3
import base64

import mlflow

# kinesis_client = boto3.client('kinesis')

def load_model(run_id):
    logged_model = f"s3://mlflow-artifacts-rmt1/0/{run_id}/artifacts/model"
    model = mlflow.pyfunc.load_model(logged_model)
    return model


class ModelService():

    def __init__(self, model):
        self.model = model


    def prepare_features(self, ride):
        features = {}
        features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
        features['trip_distance'] = ride['trip_distance']
        return features

    def predict(self, features):
        pred = self.model.predict(features)
        return float(pred[0])


    def lambda_handler(self, event):

        # print(json.dumps(event))  
        predictions_events = []
        for record in event['Records']:
            encoded_data = record['kinesis']['data']
            decoded_data = base64.b64decode(encoded_data).decode('utf-8')
            ride_event = json.loads(decoded_data)
            # print(ride_event)
            
            ride = ride_event['ride']
            ride_id = ride_event['ride_id']
            features = self.prepare_features(ride)
            prediction = self.predict(features)
        
            prediction_event = {
                'model': 'ride_duration_prediction_model',
                'version': '123',
                'prediction': {
                    'ride_duration': prediction,
                    'ride_id': ride_id   
                }
            }

            # if not TEST_RUN:
            #     kinesis_client.put_record(
            #         StreamName=PREDICTIONS_STREAM_NAME,
            #         Data=json.dumps(prediction_event),
            #         PartitionKey=str(ride_id)
            #     )
            
            predictions_events.append(prediction_event)


        return {
            'predictions': predictions_events
        }


def init(prediction_stream_name: str, run_id: str, test_run: bool):

    model = load_model(run_id)
    model_service = ModelService(model)
    return 