from sklearn.model_selection import permutation_test_score
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time
from sklearn.metrics import mean_squared_error, r2_score


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



for i, emotion in enumerate(emotions[-2:]):
    y_target = y_train.iloc[:, i]  

    svr = SVR(kernel='rbf', C=10, epsilon=0.01, gamma='scale')

    score, perm_scores, p_value = permutation_test_score(
        svr,
        X_train_scaled,
        y_target,
        scoring="r2",        
        cv=5,
        n_permutations=100,
        n_jobs=-1,
        random_state=42
    )
    print(f"## P testing for emotion:{emotion}")
    print(f"Observed R²: {score:.3f}")
    print(f"Permutation p-value: {p_value:.4f}")
    print(f"Perm Scores: {perm_scores}")
    print('\n\n')
