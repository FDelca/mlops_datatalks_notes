import requests

event = {
    "Records": [
        {
            "kinesis": {
                "kinesisSchemaVersion": "1.0",
                "partitionKey": "1",
                "sequenceNumber": "49632656867294236735995561488278562230749104803411918850",
                "data": "ewogICAgICAgICJyaWRlIjogewogICAgICAgICAgICAiUFVMb2NhdGlvbklEIjogMTMwLAogICAgICAgICAgICAiRE9Mb2NhdGlvbklEIjogMjA1LAogICAgICAgICAgICAidHJpcF9kaXN0YW5jZSI6IDMuNjYKICAgICAgICB9LCAKICAgICAgICAicmlkZV9pZCI6IDE1NgogICAgfQ==",
                "approximateArrivalTimestamp": 1661384032.549
            },
            "eventSource": "aws:kinesis",
            "eventVersion": "1.0",
            "eventID": "shardId-000000000000:49632656867294236735995561488278562230749104803411918850",
            "eventName": "aws:kinesis:record",
            "invokeIdentityArn": "arn:aws:iam::170107914870:role/lambda-kinesis-role",
            "awsRegion": "eu-west-1",
            "eventSourceARN": "arn:aws:kinesis:eu-west-1:170107914870:stream/ride-events"
        }
    ]
}

url = 'http://localhost:8080/2015-03-31/functions/function/invocations' # it comes from here https://docs.aws.amazon.com/lambda/latest/dg/images-test.html#images-test-AWSbase
response = requests.post(url, json=event)
print(response.json())
