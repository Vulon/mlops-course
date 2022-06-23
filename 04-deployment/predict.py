import pickle
import pandas as pd
import argparse
import warnings
warnings.filterwarnings(action="ignore")


CATEGORICAL_VARIABLES = ['PUlocationID', 'DOlocationID']


def load_model(model_path: str):
    with open(model_path, 'rb') as f_in:
        dv, lr = pickle.load(f_in)
    return dv, lr




def read_data(url_format: str, year: int, month: int):
    
    df = pd.read_parquet(url_format.format(year, month))
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[CATEGORICAL_VARIABLES] = df[CATEGORICAL_VARIABLES].fillna(-1).astype('int').astype('str')
    
    return df



def make_prediction(dataframe, dv, lr):
    dicts = dataframe[CATEGORICAL_VARIABLES].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)
    dataframe["prediction"] = y_pred
    return dataframe


if __name__ == "__main__":
    import os
    folder_path = os.path.dirname(__file__)
    
    parser = argparse.ArgumentParser(description='Enter year and month to run the prediction')
    parser.add_argument('--year', type=int, help='a target year')
    parser.add_argument('--month', type=int, help='a target month')
    args = parser.parse_args()
    month = args.month
    assert month < 13 and month > 0
    month = "0" + str(month) if month < 10 else str(month)
    year = args.year
    print("Loading data:")
    df = read_data('https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{}-{}.parquet', year, month)    
    
    dv, lr = load_model(os.path.join(folder_path, "model.bin"))
    df = make_prediction(df, dv, lr)
    print("Mean duration prediction", df['prediction'].mean())
