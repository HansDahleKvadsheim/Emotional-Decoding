import numpy as np
import os
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

start_time = time.time()
print("Start Time:", time.strftime("%H:%M:%S", time.localtime(start_time)))
VERBOSE = True

input_csv = 'data/full_dfc/output.csv'
feature_importance_dir = 'data/feature_importance/dfc/rfr'
os.makedirs(feature_importance_dir, exist_ok=True)

emotions = ['anticipation', 'joy', 'trust', 'fear', 'suprise', 'sadness', 'disgust', 'anger']

df = pd.read_csv(input_csv)
X = df[df.columns[:-8]]
y = df[df.columns[-8:]]

stratify_col = X['subject_no'].astype(str) + '_' + X['time_order'].astype(str)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=stratify_col
)

X_train = X_train.drop(columns=df.columns[:2])
X_test = X_test.drop(columns=df.columns[:2])

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

if VERBOSE:
    print("Training set shape:", X_train.shape, y_train.shape)
    print("Test set shape:", X_test.shape, y_test.shape)

feature_importance_dict = {}
y_pred_test_all = []
y_pred_train_all = []

print("Training one Random Forest per emotion...")

for i, emotion in enumerate(emotions):
    rf = RandomForestRegressor(
        n_estimators=2000,
        random_state=42,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        max_depth=30,
        n_jobs=4
    )
    rf.fit(X_train, y_train.iloc[:, i])
    
    y_pred_test = rf.predict(X_test)
    y_pred_train = rf.predict(X_train)
    y_pred_test_all.append(y_pred_test)
    y_pred_train_all.append(y_pred_train)

    feature_importance_dict[emotion] = rf.feature_importances_

y_pred_test_all = np.vstack(y_pred_test_all).T
y_pred_train_all = np.vstack(y_pred_train_all).T

print("RFR Model Performance:")
print("Test MSE:", mean_squared_error(y_test, y_pred_test_all))
print("Test R²:", r2_score(y_test, y_pred_test_all, multioutput='uniform_average'))

print("Train MSE:", mean_squared_error(y_train, y_pred_train_all))
print("Train R²:", r2_score(y_train, y_pred_train_all, multioutput='uniform_average'))

feature_importance_df = pd.DataFrame(
    data=[feature_importance_dict[e] for e in emotions],
    index=emotions,
    columns=[f"Feature_{i}" for i in range(X_train.shape[1])]
)
feature_importance_df.to_csv(os.path.join(feature_importance_dir, 'importance.csv'))

end_time = time.time()
print("End Time:", time.strftime("%H:%M:%S", time.localtime(end_time)))
print(f"Elapsed Time: {end_time - start_time:.2f} seconds")

