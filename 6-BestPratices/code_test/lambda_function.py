import os
import model



PREDICTIONS_STREAM_NAME = os.getenv('PREDICTIONS_STREAM_NAME', 'ride-predictions')
RUN_ID = os.getenv('RUN_ID', 'b24cae9f5daa423f80f4e0bd4435e72d')
TEST_RUN = os.getenv('TEST_RUN', 'False') == 'True'

model_service = model.init(
    prediction_stream_name=PREDICTIONS_STREAM_NAME,
    run_id=RUN_ID,
    test_run=TEST_RUN,
)

def lambda_handler(event, context):
    return model_service.lambda_handler(event)
