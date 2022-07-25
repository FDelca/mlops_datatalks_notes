# First we should define the basic image
FROM python:3.8.12-slim

# Define a working directory inside our docker image - (similar to cd command)
WORKDIR /app
# Copy the files that we need, the last argument in the list
# referes to the place were the files are going to be copy to
COPY ["requirements.txt", "./"]

# Install the requirements used in the development stage
RUN pip install -r requirements.txt

# Copy other dependencies for our image
COPY ["predict_service.py", "model_C=0.5.bin", "./"]

# EXPOSE THE PORT TO BE CONSUMED BY THE HOST (YOUR COMPUTER)
EXPOSE 9696

# SET THE ENTRYPOINT AS WELL
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict_service:app"]
