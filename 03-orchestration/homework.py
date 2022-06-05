import pandas as pd
from prefect import task, flow
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from prefect import get_run_logger
import os
import datetime
from prefect.flow_runners import SubprocessFlowRunner
import pickle



@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    logger = get_run_logger()
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:        
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):
    logger = get_run_logger()
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return mse


@task
def get_paths(date: str, file_pattern = "fhv_tripdata_{}.parquet"):
    logger = get_run_logger()
    def get_month_start(datetime_object: datetime.datetime):
        start_date = datetime.datetime(datetime_object.year, datetime_object.month, 1)
        return start_date

    module_root = os.path.dirname(__file__)
    folder_path = os.path.join( module_root, "data/raw" )
    
    if date:
        current_date = datetime.datetime.strptime(date, "%Y-%m-%d")
    else:
        current_date = datetime.datetime.now()
    month_start = get_month_start(current_date)
    val_month = get_month_start(month_start - datetime.timedelta(2))
    train_month = get_month_start(val_month - datetime.timedelta(2))
    val_path = os.path.join(folder_path, file_pattern.format(val_month.strftime("%Y-%m"))) 
    train_path = os.path.join(folder_path, file_pattern.format(train_month.strftime("%Y-%m"))) 
    logger.info( f" input date {date}. validation path {val_path}, train path {train_path} " )
    return train_path, val_path


def save_files(model, vectorizer, date: str):
    module_root = os.path.dirname(__file__)
    folder_path = os.path.join( module_root, "data/binary" )
    os.makedirs(folder_path, exist_ok=True)
    with open( os.path.join(folder_path, f"model-{date}.bin"), 'wb' ) as file:
        pickle.dump(model, file)
    with open( os.path.join(folder_path, f"dv-{date}.b"), 'wb' ) as file:
        pickle.dump(vectorizer, file)
    


@flow
def main(date:str =None):
    categorical = ['PUlocationID', 'DOlocationID']
    train_path, val_path = get_paths(date).result()

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, train=False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    val_mse = run_model(df_val_processed, categorical, dv, lr)
    save_files(lr, dv, date)



# date = "2021-08-15"
# main(date=date)
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule


DeploymentSpec(
    flow=main,
    name="scheduled_taxi_lr",
    schedule=CronSchedule(cron="0 9 15 * *"),
    flow_runner=SubprocessFlowRunner()
)
