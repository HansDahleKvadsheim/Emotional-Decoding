import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from joblib import Parallel, delayed
from sklearn.base import clone

start_time = time.time()
print("Start Time:", time.strftime("%H:%M:%S", time.localtime(start_time)))

data_path = 'data/full_dfc/output.csv'
emotions = ['anticipation', 'joy', 'trust', 'fear', 'suprise', 'sadness', 'disgust', 'anger']

df = pd.read_csv(data_path)
X = df[df.columns[:-8]]
y = df[df.columns[-8:]]

stratify_col = X['subject_no'].astype(str) + '_' + X['time_order'].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=stratify_col
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf_template = RandomForestRegressor(
    n_estimators=2000,
    random_state=42,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    max_depth=30,
    n_jobs=1   
)

y_pred_test_all = []
y_pred_train_all = []

for i, emotion in enumerate(emotions):
    print(f"\nTraining Random Forest for emotion: {emotion}")
    rf = clone(rf_template)
    rf.fit(X_train, y_train.iloc[:, i])
    y_pred_test_all.append(rf.predict(X_test))
    y_pred_train_all.append(rf.predict(X_train))

y_pred_test_all = np.vstack(y_pred_test_all).T
y_pred_train_all = np.vstack(y_pred_train_all).T

mse_test_real = mean_squared_error(y_test, y_pred_test_all)
r2_test_real = r2_score(y_test, y_pred_test_all, multioutput='uniform_average')
mse_train_real = mean_squared_error(y_train, y_pred_train_all)
r2_train_real = r2_score(y_train, y_pred_train_all, multioutput='uniform_average')

print("\n### Random Forest Model Performance ###")
print(f"Test MSE: {mse_test_real:.4f}")
print(f"Test R²: {r2_test_real:.4f}")
print(f"Train MSE: {mse_train_real:.4f}")
print(f"Train R²: {r2_train_real:.4f}")

def run_permutation(seed):
    y_train_perm = y_train.sample(frac=1.0, replace=False, random_state=seed).reset_index(drop=True)
    y_pred_test_perm = []

    for i in range(len(emotions)):
        rf = clone(rf_template)
        rf.fit(X_train, y_train_perm.iloc[:, i])
        y_pred = rf.predict(X_test)
        y_pred_test_perm.append(y_pred)

    y_pred_test_perm = np.vstack(y_pred_test_perm).T
    mse_perm = mean_squared_error(y_test, y_pred_test_perm)
    r2_perm = r2_score(y_test, y_pred_test_perm, multioutput='uniform_average')
    return mse_perm, r2_perm

n_permutations = 100
print("\nStarting permutation testing...")

results = Parallel(n_jobs=-1)(delayed(run_permutation)(seed) for seed in range(n_permutations))

permuted_mse, permuted_r2 = zip(*results)
permuted_mse = np.array(permuted_mse)
permuted_r2 = np.array(permuted_r2)

print("\n### Permutation Test Results ###")
print(f"Mean Permuted Test MSE: {permuted_mse.mean():.4f} (Real: {mse_test_real:.4f})")
print(f"Mean Permuted Test R²: {permuted_r2.mean():.4f} (Real: {r2_test_real:.4f})")

p_value_r2 = np.mean(permuted_r2 >= r2_test_real)
print(f"Empirical p-value for R²: {p_value_r2:.4f}")

end_time = time.time()
print("End Time:", time.strftime("%H:%M:%S", time.localtime(end_time)))
print(f"Elapsed Time: {end_time - start_time:.2f} seconds")
