import pickle
import mlflow

from flask import Flask, request, jsonify


# # If the tracking server is running
# RUN_ID = '3b590dd0c5b2432f80b069b99c1426da'
# MLFLOW_TRACKING_URI='http://127.0.0.1:8000'
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# # Get the model from mlflow - This code is copy from mlflow ui
# logged_model = f'runs:/{RUN_ID}/model'

# If the tracking server is NOT running
RUN_ID = '3b590dd0c5b2432f80b069b99c1426da'
logged_model = f"./mlruns/1/{RUN_ID}/artifacts/model"


model = mlflow.pyfunc.load_model(logged_model)

# Data Transformation
def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID']) # This approach will turn everything into a string
    features['trip_distance'] = ride['trip_distance']
    return features


# Create a function to predict
def predict(features):
    preds = model.predict(features)
    return float(preds[0])



# Create a flask application
app = Flask('duration-prediction')

# Create a function to wrap everything
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'duration': pred,
        'model_version': RUN_ID
    }

    return jsonify(result)


# And finally run the flask application
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)