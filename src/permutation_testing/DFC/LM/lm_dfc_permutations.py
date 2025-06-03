from sklearn.model_selection import permutation_test_score
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
import pandas as pd


emotions = ['anticipation', 'joy', 'trust', 'fear', 'suprise', 'sadness', 'disgust', 'anger']



df = pd.read_csv('data/full_dfc/output.csv')

X = df[df.columns[:-8]]
y = df[df.columns[-8:]]

stratify_col = X['subject_no'].astype(str) + '_' + X['time_order'].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=stratify_col
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)







model = MultiOutputRegressor(LinearRegression())
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
true_score = r2_score(y_test, y_pred, multioutput='uniform_average')

n_permutations = 100
perm_scores = []

for _ in range(n_permutations):
    y_perm = np.random.permutation(y_train)
    model.fit(X_train, y_perm)
    y_perm_pred = model.predict(X_test)
    score = r2_score(y_test, y_perm_pred, multioutput='uniform_average')
    perm_scores.append(score)

p_val = (np.sum(np.array(perm_scores) >= true_score) + 1) / (n_permutations + 1)

print(f"## LINEAR MODEL ##")
print("p-value:", p_val)
print(f"p-scores for permuations: {perm_scores}")
print("\n\n")

model = MultiOutputRegressor(Lasso(alpha=0.0006551285568595509))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
true_score = r2_score(y_test, y_pred, multioutput='uniform_average')

perm_scores = []

for _ in range(n_permutations):
    y_perm = np.random.permutation(y_train)
    model.fit(X_train, y_perm)
    y_perm_pred = model.predict(X_test)
    score = r2_score(y_test, y_perm_pred, multioutput='uniform_average')
    perm_scores.append(score)

p_val = (np.sum(np.array(perm_scores) >= true_score) + 1) / (n_permutations + 1)

print(f"## LASSO MODEL ##")
print("p-value:", p_val)
print(f"p-scores for permuations: {perm_scores}")
print("\n\n")

model = MultiOutputRegressor(Ridge(alpha=3237.45754281764))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
true_score = r2_score(y_test, y_pred, multioutput='uniform_average')

perm_scores = []

for _ in range(n_permutations):
    y_perm = np.random.permutation(y_train)
    model.fit(X_train, y_perm)
    y_perm_pred = model.predict(X_test)
    score = r2_score(y_test, y_perm_pred, multioutput='uniform_average')
    perm_scores.append(score)

p_val = (np.sum(np.array(perm_scores) >= true_score) + 1) / (n_permutations + 1)

print(f"## RIDGE MODEL ##")
print("p-value:", p_val)
print(f"p-scores for permuations: {perm_scores}")
print("\n\n")
