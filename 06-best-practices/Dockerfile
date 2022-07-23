FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "batch.py", "batch.py" ]
COPY [ "model.bin", "model.bin" ]


ENV INPUT_FILE_PATTERN="s3://nyc-duration/in/fhv_tripdata_{year:04d}-{month:02d}.parquet"
ENV OUTPUT_FILE_PATTERN="s3://nyc-duration/out/fhv_tripdata_{year:04d}-{month:02d}.parquet"
ENV S3_ENDPOINT_URL="http://127.0.0.1:4566"

ENTRYPOINT [ "python", "batch.py" ]