import numpy as np
import os
import pandas as pd
import scipy.io
from sklearn.preprocessing import minmax_scale, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso, Ridge
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import time


df = pd.read_csv('data/roi/all_networks.csv')

X = df[df.columns[:-8]]
y = df[df.columns[-8:]]

scaler = StandardScaler()


stratify_col = X['subject_no'].astype(str) + '_' + X['time_quantization'].astype(str)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,stratify=stratify_col)


X_train = X_train.iloc[:, :400]
X_test = X_test.iloc[:, :400]

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

start_time = time.time()
print("Start Time, Hyperparameter Search:", time.strftime("%H:%M:%S", time.localtime(start_time)))

lasso = Lasso(random_state=42)
ridge = Ridge(random_state=42)

lasso_param_grid = {"alpha": np.logspace(-4, 4, 50)}
ridge_param_grid = {"alpha": np.logspace(-4, 4, 50)}

grid_search_lasso = GridSearchCV(lasso, lasso_param_grid, cv=5, scoring="r2", n_jobs=-1, verbose=2)
grid_search_ridge = GridSearchCV(ridge, ridge_param_grid, cv=5, scoring="r2", n_jobs=-1, verbose=2)

grid_search_lasso.fit(X_train, y_train)
grid_search_ridge.fit(X_train, y_train)

best_lasso = grid_search_lasso.best_estimator_
best_ridge = grid_search_ridge.best_estimator_

y_pred_lasso = best_lasso.predict(X_test)
y_pred_ridge = best_ridge.predict(X_test)

y_pred_train_lasso = best_lasso.predict(X_train)
y_pred_train_ridge = best_ridge.predict(X_train)

mse_test_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_test_lasso = r2_score(y_test, y_pred_lasso, multioutput='uniform_average')
mse_train_lasso = mean_squared_error(y_train, y_pred_train_lasso)
r2_train_lasso = r2_score(y_train, y_pred_train_lasso, multioutput='uniform_average')

mse_test_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_test_ridge = r2_score(y_test, y_pred_ridge, multioutput='uniform_average')
mse_train_ridge = mean_squared_error(y_train, y_pred_train_ridge)
r2_train_ridge = r2_score(y_train, y_pred_train_ridge, multioutput='uniform_average')

print("Lasso Best Alpha:", grid_search_lasso.best_params_)
print("Lasso Test MSE:", mse_test_lasso)
print("Lasso Test R^2:", r2_test_lasso)
print("Lasso Train MSE:", mse_train_lasso)
print("Lasso Train R^2:", r2_train_lasso)

print("Ridge Best Alpha:", grid_search_ridge.best_params_)
print("Ridge Test MSE:", mse_test_ridge)
print("Ridge Test R^2:", r2_test_ridge)
print("Ridge Train MSE:", mse_train_ridge)
print("Ridge Train R^2:", r2_train_ridge)

end_time = time.time()
print("End Time, Hyperparameter Search:", time.strftime("%H:%M:%S", time.localtime(end_time)))
print("Elapsed Time:", round(end_time - start_time, 2), "seconds")


