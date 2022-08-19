import json
import requests

from deepdiff import DeepDiff

event = {
    "Records": [
        {
            "kinesis": {
                "kinesisSchemaVersion": "1.0",
                "partitionKey": "1",
                "sequenceNumber": "49631783078782244062766649846789567042518784522594549762",
                "data": "ewogICAgICAgICJyaWRlIjogewogICAgICAgICAgICAiUFVMb2NhdGlvbklEIjogMTMwLAogICAgICAgICAgICAiRE9Mb2NhdGlvbklEIjogMjA1LAogICAgICAgICAgICAidHJpcF9kaXN0YW5jZSI6IDMuNjYKICAgICAgICB9LCAKICAgICAgICAicmlkZV9pZCI6IDE1NgogICAgfQ==",
                "approximateArrivalTimestamp": 1658933550.303
            },
            "eventSource": "aws:kinesis",
            "eventVersion": "1.0",
            "eventID": "shardId-000000000000:49631783078782244062766649846789567042518784522594549762",
            "eventName": "aws:kinesis:record",
            "invokeIdentityArn": "arn:aws:iam::170107914870:role/lambda-kinesis-role",
            "awsRegion": "us-east-1",
            "eventSourceARN": "arn:aws:kinesis:us-east-1:170107914870:stream/ride-events"
        }
    ]
}

url = 'http://localhost:8080/2015-03-31/functions/function/invocations' # in the README.md file
actual_response = requests.post(url, json=event).json()
expected_response = {
    'predictions': [{
        'model': 'ride_duration_prediction_model', 
        'version': 'b24cae9f5daa423f80f4e0bd4435e72d', 
        'prediction': {
            'ride_duration': 18.2, 
            'ride_id': 156
            }
            }]}

print(f"actual response: {json.dumps(actual_response, indent=2)}")

diff = DeepDiff(actual_response, expected_response)
print("diff: {diff}")
assert "values_changed" not in diff