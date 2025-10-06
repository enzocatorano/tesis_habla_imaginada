# notebooks/baseline_classifiers.py
import numpy as np
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

feat_dir = Path("features/preproc_dwt_L5_db4/per_subject")
subj = "S01_EEG"
X3 = np.load(feat_dir / f"{subj}_features.npy")
labels = np.load(feat_dir / f"{subj}_labels.npy")
task = np.load(feat_dir / f"{subj}_task.npy")

# choose task vowels (tval=0) as example
mask = (task==0)
X = X3[mask].reshape(mask.sum(), -1)
y = labels[mask][:,1].astype(int) - 1  # vowels 1..5 -> 0..4

scaler = StandardScaler()
Xs = scaler.fit_transform(X)

cv = StratifiedKFold(5, shuffle=True, random_state=42)
print("Logistic CV (L2):", cross_val_score(LogisticRegression(max_iter=5000), Xs, y, cv=cv).mean())
print("RandomForest CV:", cross_val_score(RandomForestClassifier(n_estimators=200), X, y, cv=cv).mean())
