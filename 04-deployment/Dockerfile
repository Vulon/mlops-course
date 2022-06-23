FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

WORKDIR /app

RUN pip install -U pip
RUN pip install pipenv

COPY [ "Pipfile", "Pipfile.lock", "predict.py", "./" ]

RUN pipenv install --system --deploy
