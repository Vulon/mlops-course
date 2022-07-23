cd "$(dirname "$0")"



#docker-compose up -d

sleep 5

TEST_FOLDER_NAME=$(dirname $(realpath "$0"))

PROJECT_FOLDER_PATH=$(dirname "$TEST_FOLDER_NAME")
PROJECT_FOLDER_PATH=$(dirname "$PROJECT_FOLDER_PATH")

echo $PROJECT_FOLDER_PATH

aws s3 mb s3://nyc-duration --endpoint-url=http://localhost:4566

aws s3 cp C:/PythonProjects/mlops-course/06-best-practices/data --endpoint-url=http://localhost:4566 s3://nyc-duration --recursive


export INPUT_FILE_PATTERN="s3://nyc-duration/in/fhv_tripdata_{year:04d}-{month:02d}.parquet"
export OUTPUT_FILE_PATTERN="s3://nyc-duration/out/fhv_tripdata_{year:04d}-{month:02d}.parquet"
export S3_ENDPOINT_URL="http://127.0.0.1:4566"