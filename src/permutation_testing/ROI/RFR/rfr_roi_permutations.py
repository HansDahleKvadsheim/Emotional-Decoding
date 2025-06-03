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


df = pd.read_csv('data/roi/all_networks.csv')

X = df[df.columns[:-8]]
y = df[df.columns[-8:]]

stratify_col = X['subject_no'].astype(str) + '_' + X['time_quantization'].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=stratify_col
)

X_train = X_train.iloc[:, :400]
X_test = X_test.iloc[:, :400]


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



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

true_r2 = r2

permuted_r2_scores = []

print("\nRunning permutation test with 100 permutations...")
for i in range(100):
    y_train_permuted = y_train.apply(np.random.permutation)

    permuted_model = RandomForestRegressor(
        n_estimators=2000,
        random_state=42,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        max_depth=None,
        n_jobs=4
    )
    permuted_model.fit(X_train, y_train_permuted)

    y_permuted_pred = permuted_model.predict(X_test)
    r2_perm = r2_score(y_test, y_permuted_pred, multioutput='uniform_average')

    permuted_r2_scores.append(r2_perm)


num_extreme = sum(r2_perm >= true_r2 for r2_perm in permuted_r2_scores)
p_value = (num_extreme + 1) / (100 + 1)

print("\nPermutation Test Results:")
print(f"True R^2 Score: {true_r2:.4f}")
print(f"Empirical p-value: p = {p_value:.4f}")
print(f"Permutation performances list {permuted_r2_scores}")

