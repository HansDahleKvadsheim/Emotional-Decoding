from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler


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
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
}

model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=4)

search = RandomizedSearchCV(model, param_grid, n_iter=20, cv=3, n_jobs=4, verbose=1)
search.fit(X_train, y_train)

best_params = search.best_params_
print("Best Params:", best_params)


