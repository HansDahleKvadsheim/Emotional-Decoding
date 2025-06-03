import numpy as np
import os
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

start_time = time.time()
print("Start Time:", time.strftime("%H:%M:%S", time.localtime(start_time)))

df = pd.read_csv('data/full_dfc/output.csv')

X = df[df.columns[:-8]]
y = df[df.columns[-8:]]

emotions = ['anticipation', 'joy', 'trust', 'fear', 'suprise', 'sadness', 'disgust', 'anger']
feature_importance_dir = 'data/feature_importance/dfc/svr/'

os.makedirs(feature_importance_dir, exist_ok=True)

stratify_col = X['subject_no'].astype(str) + '_' + X['time_order'].astype(str)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=stratify_col
)

X_train = X_train.drop(columns=df.columns[:2])
X_test = X_test.drop(columns=df.columns[:2])

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svr_model = MultiOutputRegressor(SVR(kernel='linear', C=0.01, epsilon=0.01, gamma='scale'), n_jobs=4)
svr_model.fit(X_train, y_train)

y_pred = svr_model.predict(X_test)
y_pred_train = svr_model.predict(X_train)

print("SVM Model Performance:")
print("Test MSE:", mean_squared_error(y_test, y_pred))
print("Test R²:", r2_score(y_test, y_pred, multioutput='uniform_average'))
print("Train MSE:", mean_squared_error(y_train, y_pred_train))
print("Train R²:", r2_score(y_train, y_pred_train, multioutput='uniform_average'))

feature_importance_array = np.zeros((len(emotions), X_train.shape[1]))
for i, estimator in enumerate(svr_model.estimators_):
    feature_importance_array[i, :] = estimator.coef_.flatten()

feature_importance_df = pd.DataFrame(
    data=feature_importance_array,
    index=emotions,
    columns=[f"Feature_{i}" for i in range(X_train.shape[1])]
)

feature_importance_df.to_csv(os.path.join(feature_importance_dir, 'importance.csv'))

end_time = time.time()
print("End Time:", time.strftime("%H:%M:%S", time.localtime(end_time)))
print(f"Elapsed Time: {end_time - start_time:.2f} seconds")
