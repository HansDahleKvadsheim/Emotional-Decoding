import numpy as np
import os
import pandas as pd
import scipy.io
from sklearn.preprocessing import minmax_scale, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso, Ridge
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import time

start_time = time.time()
print("Start Time:", time.strftime("%H:%M:%S", time.localtime(start_time)))
VERBOSE = True


df = pd.read_csv('/cluster/home/augustsa/demonstrasjon/roi/SalVentAttn_Vis_excluded.csv')

X = df[df.columns[:-8]]
y = df[df.columns[-8:]]

stratify_col = X['subject_no'].astype(str) + '_' + X['time_quantization'].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=stratify_col
)

X_train = X_train.iloc[:, :292]
X_test = X_test.iloc[:, :292]


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


if VERBOSE:
    print("Training set shape:", X_train.shape, y_train.shape)
    print("Test set shape:", X_test.shape, y_test.shape)


base_rf = RandomForestRegressor(n_estimators=2000, random_state=42, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', max_depth=None, n_jobs=4)

print("Training Random Forest model...")
base_rf.fit(X_train, y_train)



y_pred = base_rf.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred, multioutput='uniform_average')

print("RFR Model Performance:")
print("Test Mean Squared Error (MSE):", mse)
print("Test R^2 Score:", r2)

y_pred_train = base_rf.predict(X_train)

mse_train = mean_squared_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train, multioutput='uniform_average')

print("RFR model preformance, train: ")
print("Train mean squared Error (MSE):", mse_train)
print("Train R^2 Score:", r2_train)

end_time = time.time()
print("End Time:", time.strftime("%H:%M:%S", time.localtime(end_time)))


elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time:.2f} seconds")




