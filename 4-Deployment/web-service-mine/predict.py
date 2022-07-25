import pickle 
from flask import Flask, request, jsonify

# Load the model
with open('lin_reg.bin', 'rb') as f_in:
    (dv, model) = pickle.load(f_in)


# Data Transformation
def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID']) # This approach will turn everything into a string
    features['trip_distance'] = ride['trip_distance']
    return features


# Create a function to predict
def predict(features):
    X = dv.transform(features)
    preds = model.predict(X)
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
        'duration': pred
    }

    return jsonify(result)


# And finally run the flask application
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)