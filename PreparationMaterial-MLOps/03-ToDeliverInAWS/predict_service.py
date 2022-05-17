import pickle
from flask import Flask
from flask import request
from flask import jsonify

C = 0.5
output_file = f'model_C={C}.bin'

with open(output_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)
    
app = Flask('churn')

# We change it here to POST method to give information about the customer
@app.route('/predict', methods=['POST'])
def predict_single():
    customer = request.get_json() # It will turn into a json file into a python dictionary

    ## We should put it in a separate function ##
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    # Make decision in our service - to help the marketing team (they do not know which threshold to use)
    churn = y_pred >= 0.5
    #############################################

    # Our response it needs to be json so we need to prepare it as well
    result = {
        'churn_probability': float(y_pred), 
        'churn': bool(churn)
    }

    return jsonify(result)

# This function should live inside __main__ method
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
