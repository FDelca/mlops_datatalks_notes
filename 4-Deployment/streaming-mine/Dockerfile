FROM public.ecr.aws/lambda/python:3.9

RUN pip install -U pip

RUN pip install mlflow==1.26.1
RUN pip install boto3==1.24.11
RUN pip install scikit-learn==1.1.0

COPY ["lambda_function.py", "mlruns", "mlflow.db","./"]

CMD ["lambda_function.lambda_handler"]