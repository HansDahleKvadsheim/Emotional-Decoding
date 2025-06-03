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





df = pd.read_csv('data/full_dfc/output.csv')

X = df[df.columns[:-8]] 
y = df[df.columns[-8:]]

scaler = StandardScaler()

emotions = ['anticipation', 'joy', 'trust', 'fear', 'suprise', 'sadness', 'disgust', 'anger']
feature_importance_dir = '/data/feature_importance/dfc'


stratify_col = X['subject_no'].astype(str) + '_' + X['time_order'].astype(str)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,stratify=stratify_col)


X_train = X_train.drop(columns=df.columns[:2])
X_test = X_test.drop(columns=df.columns[:2])

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

start_time = time.time()


### LINEAR MODEL

print("Start Time, Linear:", time.strftime("%H:%M:%S", time.localtime(start_time)))

linear_model = LinearRegression(n_jobs=4)

linear_model.fit(X_train, y_train)


y_pred = linear_model.predict(X_test)
y_pred_train = linear_model.predict(X_train)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred, multioutput='uniform_average')

print("Linear Model Performance:")
print("Test Mean Squared Error (MSE):", mse)
print("Test R^2 Score:", r2)

print("\n")


mse_train = mean_squared_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train, multioutput='uniform_average')

print("Linear model preformance, train: ")
print("Train mean squared Error (MSE):", mse_train)
print("Train R^2 Score:", r2_train)


feature_importance = pd.DataFrame(data=linear_model.coef_.T, columns=emotions)
feature_importance.to_csv(path_or_buf=os.path.join(feature_importance_dir, 'lm/importance.csv'))

end_time = time.time()
print("End Time:", time.strftime("%H:%M:%S", time.localtime(end_time)))


elapsed_time = end_time - start_time
print(f"Elapsed Time, Linear: {elapsed_time:.2f} seconds")

print("\n\n")



### LASSO MODEL

start_time = time.time()
print("Start Time, Lasso:", time.strftime("%H:%M:%S", time.localtime(start_time)))

lasso_model = Lasso(alpha=0.0006551285568595509, random_state=42)

lasso_model.fit(X_train, y_train)


y_pred = lasso_model.predict(X_test)
y_pred_train = lasso_model.predict(X_train)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred, multioutput='uniform_average')

print("Lasso Model Performance:")
print("Test Mean Squared Error (MSE):", mse)
print("Test R^2 Score:", r2)



mse_train = mean_squared_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train, multioutput='uniform_average')

print("Lasso Model preformance, train: ")
print("Train mean squared Error (MSE):", mse_train)
print("Train R^2 Score:", r2_train)

feature_importance_lasso = pd.DataFrame(data=lasso_model.coef_.T, columns=emotions)
feature_importance_lasso.to_csv(path_or_buf=os.path.join(feature_importance_dir, 'lasso/importance.csv'))

end_time = time.time()
print("End Time:", time.strftime("%H:%M:%S", time.localtime(end_time)))


elapsed_time = end_time - start_time
print(f"Elapsed Time, Lasso: {elapsed_time:.2f} seconds")

print("\n\n")


### RIDGE MODEL

start_time = time.time()
print("Start Time, Ridge:", time.strftime("%H:%M:%S", time.localtime(start_time)))

ridge_model = Ridge(alpha=3237.45754281764, random_state=42)

ridge_model.fit(X_train, y_train)


y_pred = ridge_model.predict(X_test)
y_pred_train = ridge_model.predict(X_train)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred, multioutput='uniform_average')

print("Ridge model Performance:")
print("Test Mean Squared Error (MSE):", mse)
print("Test R^2 Score:", r2)



mse_train = mean_squared_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train, multioutput='uniform_average')

print("Ridge Model preformance, train: ")
print("Train mean squared Error (MSE):", mse_train)
print("Train R^2 Score:", r2_train)



feature_importance_ridge = pd.DataFrame(data=ridge_model.coef_.T, columns=emotions)
feature_importance_ridge.to_csv(path_or_buf=os.path.join(feature_importance_dir, 'ridge/importance.csv'))


end_time = time.time()
print("End Time:", time.strftime("%H:%M:%S", time.localtime(end_time)))


elapsed_time = end_time - start_time
print(f"Elapsed Time, Lasso: {elapsed_time:.2f} seconds")



