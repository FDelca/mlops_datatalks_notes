from flask import Flask

app = Flask('ping')


@app.route('/ping', methods=['GET'])
def ping():
    return "Pong"
    
# This function should live inside __main__ method
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
