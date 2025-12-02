import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('../pasctraining.csv')
le = LabelEncoder()
X = df.drop(columns=['Class'])
Y = le.fit_transform(df['Class'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0, stratify=Y)

pasc_idx = (Y_train == 0)
control_idx = (Y_train == 1)
X_pasc = X_train.iloc[pasc_idx]
Y_pasc = Y_train[pasc_idx]
X_control = X_train.iloc[control_idx]
Y_control = Y_train[control_idx]

accs = []
prop = .9
B = 100
n = len(Y_control)
for _ in range(B):
    print(f"Bootstrap iteration {_+1}/{B}")
    idx = np.random.randint(0, n, n)
    X_control_b = X_control.iloc[idx]
    Y_control_b = Y_control[idx]
    Xb = pd.concat([X_pasc, X_control_b], axis=0)
    Yb = np.concatenate([Y_pasc, Y_control_b], axis=0)
    
    RF = RandomForestClassifier(n_estimators=100, random_state=0)
    RF.fit(Xb, Yb)
    pred = RF.predict(X_test)
    accs.append((pred == Y_test).mean())

std = np.std(accs)
print(std)
lower = np.percentile(accs, 2.5)
upper = np.percentile(accs, 97.5)
print(f"CI: {lower:.3f} - {upper:.3f}")