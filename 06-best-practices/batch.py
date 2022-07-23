#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd
import os
import argparse


def get_input_path(year, month):
    default_input_pattern = 'https://raw.githubusercontent.com/alexeygrigorev/datasets/master/nyc-tlc/fhv/fhv_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    default_output_pattern = 'taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)


def read_data(filename: str):
    options = {
        'client_kwargs': {
            'endpoint_url': os.environ.get("S3_ENDPOINT_URL")
        }
    }
    print("Trying to read file", filename)
    print("Connection options", options)
    df = pd.read_parquet(filename, storage_options=options)
        
    return df

def prepare_data(df: pd.DataFrame, categorical_features: list):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical_features] = df[categorical_features].fillna(-1).astype('int').astype('str')
    
    return df


def read_model(filepath: str):
    with open(filepath, 'rb') as f_in:
        dv, lr = pickle.load(f_in)
    return dv, lr


def main(year: int, month: int, model_path: str):
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)
    print(input_file, output_file)
    categorical = ['PUlocationID', 'DOlocationID']
    df = read_data(input_file)
    df = prepare_data(df, categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    dv, lr = read_model(model_path)

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)
    print('predicted mean duration:', y_pred.mean())

    df_result = pd.DataFrame([df['ride_id'], y_pred], columns=["ride_id", "predicted_duration"])
    df_result = pd.DataFrame({"ride_id": df['ride_id'], "predicted_duration": y_pred})
    options = {
        'client_kwargs': {
            'endpoint_url': os.environ.get("S3_ENDPOINT_URL")
        }
    }
    print("Predicted sum", y_pred.sum())
    print(df_result.head())
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Score taxi data')
    parser.add_argument('--year', type=int, help='year of the data')
    parser.add_argument('--month', type=int, help='month of the data')

    args = parser.parse_args()
    main(args.year, args.month, "model.bin")