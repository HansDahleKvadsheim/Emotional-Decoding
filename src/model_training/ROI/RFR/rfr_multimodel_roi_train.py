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

data_path = 'data/roi/all_networks.csv'
output_dir = 'importances/roi/rfr/'
os.makedirs(output_dir, exist_ok=True)

emotions = ['anticipation', 'joy', 'trust', 'fear', 'suprise', 'sadness', 'disgust', 'anger']

df = pd.read_csv(data_path)
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

feature_importances = []
y_pred_test_all = []
y_pred_train_all = []

for i, emotion in enumerate(emotions):
    print(f"\nTraining Random Forest for emotion: {emotion}")

    rf = RandomForestRegressor(
        n_estimators=2000,
        random_state=42,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        max_depth=None,
        n_jobs=4
    )

    rf.fit(X_train, y_train.iloc[:, i])

    y_pred_test = rf.predict(X_test)
    y_pred_train = rf.predict(X_train)

    y_pred_test_all.append(y_pred_test)
    y_pred_train_all.append(y_pred_train)

    importance = rf.feature_importances_
    feature_importances.append(importance)

    fi_df = pd.DataFrame({
        'feature_index': np.arange(len(importance)),
        'importance': importance
    })
    fi_df.to_csv(os.path.join(output_dir, f'{emotion}_importance.csv'), index=False)

y_pred_test_all = np.vstack(y_pred_test_all).T
y_pred_train_all = np.vstack(y_pred_train_all).T

mse_test = mean_squared_error(y_test, y_pred_test_all)
r2_test = r2_score(y_test, y_pred_test_all, multioutput='uniform_average')

mse_train = mean_squared_error(y_train, y_pred_train_all)
r2_train = r2_score(y_train, y_pred_train_all, multioutput='uniform_average')

print("\n=== Random Forest Model Performance ===")
print(f"Test MSE: {mse_test:.4f}")
print(f"Test R²: {r2_test:.4f}")
print(f"Train MSE: {mse_train:.4f}")
print(f"Train R²: {r2_train:.4f}")

feature_importances = np.array(feature_importances) # shape: [n_emotions, n_features]
importance_df = pd.DataFrame(feature_importances, index=emotions)
importance_df.to_csv(os.path.join(output_dir, 'all_emotions_feature_importance.csv'))

end_time = time.time()
print("End Time:", time.strftime("%H:%M:%S", time.localtime(end_time)))
print(f"Elapsed Time: {end_time - start_time:.2f} seconds")
