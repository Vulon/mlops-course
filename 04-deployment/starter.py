#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pickle
import pandas as pd


# In[4]:


with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)


# In[5]:


categorical = ['PUlocationID', 'DOlocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[7]:


df = read_data('https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_2021-02.parquet')


# In[28]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = lr.predict(X_val)
df["prediction"] = y_pred
df['ride_id'] = df["pickup_datetime"].dt.strftime("%Y/%m_") + df.index.astype("str")


# In[29]:


df[["ride_id", "prediction"]].to_parquet(
    "data/fhv_tripdata_2021-02_prediction.parquet",
    engine='pyarrow',
    compression=None,
    index=False
)


# In[ ]:




