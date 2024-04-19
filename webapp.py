import streamlit as st
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from scipy.stats._boost import nct_ufunc

# Load the XGBoost model
booster = xgb.Booster()
booster.load_model('xgb_model.bin')

# Load the KMeans and OneHotEncoder models
with open('kmeans.pkl', 'rb') as file:
    kmeans = pickle.load(file)
with open('encoder.pkl', 'rb') as file:
    enc = pickle.load(file)

def inferences(pickup_longitude, pickup_latitude, tpep_pickup_datetime):
    df_predict = pd.DataFrame(data=[[pickup_latitude, pickup_longitude]], columns=["pickup_latitude", "pickup_longitude"])
    df_predict['cluster_label'] = kmeans.predict(df_predict)
    
    df_predict.insert(loc=0, column='tpep_pickup_datetime', value=[tpep_pickup_datetime])
    df_predict["tpep_pickup_datetime"] = df_predict["tpep_pickup_datetime"].apply(pd.to_datetime)
    df_predict['pickup_hour'] = df_predict["tpep_pickup_datetime"].dt.hour
    df_predict['pickup_dayofweek'] = df_predict["tpep_pickup_datetime"].dt.dayofweek

    cat_columns = ['pickup_dayofweek', 'pickup_hour', 'cluster_label']
    datastruct_predict_encoded = enc.transform(df_predict[cat_columns]).toarray()
    df_predict_encoded_df = pd.DataFrame(datastruct_predict_encoded, columns=enc.get_feature_names(cat_columns))
    df_predict_model = pd.concat([df_predict.reset_index(), df_predict_encoded_df], axis=1).drop(['index'], axis=1)

    df_predict_model = df_predict_model.drop(cat_columns, axis=1)
    
    # drop additional columns here
    cols_to_remove = ['pickup_dayofweek_0', 'pickup_dayofweek_1', 'pickup_dayofweek_2', 'pickup_dayofweek_3', 'pickup_dayofweek_4', 'pickup_dayofweek_5', 'pickup_dayofweek_6']
    df_predict_model = df_predict_model.drop(["pickup_latitude", "pickup_longitude" , "tpep_pickup_datetime"], axis=1)
    df_predict_model = df_predict_model.drop(cols_to_remove, axis=1)
    
    df_predict_model[df_predict_model.columns] = df_predict_model[df_predict_model.columns].astype(int)
    
    # Convert DataFrame to DMatrix
    dmatrix = xgb.DMatrix(df_predict_model)

    # Use XGBoost model to predict
    prediction = booster.predict(dmatrix)
    prediction = round(prediction[0]*100, 2)
    return prediction

def main():
    st.title("Demand Prediction App")

    pickup_longitude = st.text_input("Enter pickup longitude:")
    pickup_latitude = st.text_input("Enter pickup latitude:")
    tpep_pickup_datetime = st.text_input("Enter pickup date and time (YYYY-MM-DD HH:MM:SS):", value="2020-02-13 23:40:00")
    tpep_pickup_datetime = datetime.strptime(tpep_pickup_datetime, "%Y-%m-%d %H:%M:%S")

    if pickup_longitude and pickup_latitude and tpep_pickup_datetime:
        prediction = inferences(pickup_longitude, pickup_latitude, tpep_pickup_datetime)
        st.success(f"The demand percentage based on location and time is {prediction}%")

if __name__ == "__main__":
    main()
