FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

RUN pip install pipenv

COPY ["Pipfile", "Pipfile.lock", "score.py","./"]

RUN pipenv install --deploy --system

CMD ["python", "./score.py", "2021", "4"]
