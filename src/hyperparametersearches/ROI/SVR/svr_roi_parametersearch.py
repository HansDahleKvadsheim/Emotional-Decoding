import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.preprocessing import StandardScaler

def custom_scorer(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred, multioutput="uniform_average")
    print(f"    Fold MSE: {mse:.4f}, Fold R²: {r2:.4f}")
    return r2

scorer = make_scorer(custom_scorer, greater_is_better=True)

df = pd.read_csv('data/roi/all_networks.csv')

X = df[df.columns[:-8]]
y = df[df.columns[-8:]]

stratify_col = X['subject_no'].astype(str) + '_' + X['time_quantization'].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify_col
)

X_train = X_train.iloc[:, :400]
X_test = X_test.iloc[:, :400]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

param_grid = {
    'estimator__kernel': ['linear'],
    'estimator__C': [0.01, 0.1, 1, 10],
    'estimator__epsilon': [0.01, 0.1, 0.5],
}

base_svr = SVR(max_iter=2000)

multi_svr = MultiOutputRegressor(base_svr, n_jobs=1)

grid_search = GridSearchCV(
    estimator=multi_svr,
    param_grid=param_grid,
    scoring=scorer,
    cv=3,
    n_jobs=16,  
    verbose=2,
    error_score='nan'
)

start_time = time.time()
print("Hyperparameter tuning started at:", time.strftime("%H:%M:%S", time.localtime(start_time)))

grid_search.fit(X_train, y_train)

end_time = time.time()
print("Hyperparameter tuning finished at:", time.strftime("%H:%M:%S", time.localtime(end_time)))
print(f"Elapsed Time: {end_time - start_time:.2f} seconds")

print("Best parameters found: ", grid_search.best_params_)
print("Best R^2 Score on training data:", grid_search.best_score_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_pred_train = best_model.predict(X_train)

mse_test = mean_squared_error(y_test, y_pred)
r2_test = r2_score(y_test, y_pred, multioutput='uniform_average')
mse_train = mean_squared_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train, multioutput='uniform_average')

print("\nOptimized SVR Model Performance:")
print(f"Test MSE: {mse_test:.4f}")
print(f"Test R^2 Score: {r2_test:.4f}")
print(f"Train MSE: {mse_train:.4f}")
print(f"Train R^2 Score: {r2_train:.4f}")

