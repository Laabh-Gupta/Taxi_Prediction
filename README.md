
### Data Loading and Preprocessing: This section handles the loading and preprocessing of the dataset, including handling missing values, encoding categorical features, and splitting the data into training and testing sets.
### GPU Utilization Clustering: In this section, the K-Means algorithm is used to cluster the GPU utilization data. The optimal number of clusters is determined using the Elbow method.
### Value Prediction: An XGBoost regressor is trained on the preprocessed data to predict the target variable (value) based on the features (longitude, latitude, and datetime).
### Model Evaluation: The performance of the XGBoost regressor is evaluated using appropriate metrics, such as mean squared error (MSE) or R-squared.
### Model Deployment: The trained XGBoost regressor, encoder, and K-Means models are saved as binary (xgb_model.bin), pickle (encoder.pkl), and pickle (kmeans.pkl) files, respectively, for deployment using Streamlit.

### requirements.txt mentions required versions for packages which correlates to kaggle's versions of same packages

## To run the Streamlit application, execute the following command:
streamlit run app.py
