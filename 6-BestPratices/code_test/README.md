- Install pipenv environment: `pipenv install boto3 mlflow scikit-learn==1.1.0 --python==3.9`;
- Since we are working inside a virtualenvironment you must run: `export PIPENV_IGNORE_VIRTUALENVS=1` - so it installs the environment anyway; 
- Install environment: `pipenv install`
- Install pytest only for development environment: `pipenv install --dev pytest`;
- Select the python interpert in vscode
    - To know its location run `pipenv --venv`
    - And selected inside vscode
- It will appear a symbol to do pytests in vscode
- Activate the environment: `pipenv shell`
- And check if we do have `pytest` installed: `which pytest`

- Create a directory to perform the tests `mkdir tests`
- And configure a `pytest` in vscode
- Create two files inside `tests` folder:
    1. `model_test.py` with the test
    2. `__init__.py` so python now it is runnable

- Check if the model is working:
    ```python
    import mlflow
    RUN_ID = 'b24cae9f5daa423f80f4e0bd4435e72d'
    logged_model = f"s3://mlflow-artifacts-rmt1/0/{RUN_ID}/artifacts/model"
    model = mlflow.pyfunc.load_model(logged_model)
    ```
    - There was the need to install `psutil`: `pipenv install psutil`


- Test it inside `Docker`:
    ```bash
    docker build -t stream-model-duration:v2 .
    ```

    ```bash
    docker run -it --rm \
        -p 8080:8080 \
        -e PREDICTIONS_STREAM_NAME="ride-predictions" \
        -e RUN_ID="b24cae9f5daa423f80f4e0bd4435e72d" \
        -e TEST_RUN="True" \
        -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
        -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
        -e AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION}" \
        stream-model-duration:v1
    ```



- To run all the tests build in the command line:
    ```bash
    pipenv run pytest tests/
    ```

- Tests must be fast, so we should not load any model, for testing its function we should create a mock model

- Add `callbacks` - which are things that will be invoke after each predictions, put it something into kinesis stream it will be one of them.