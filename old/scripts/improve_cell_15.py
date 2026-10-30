# 15. Temporal smoothing + validation-chosen threshold (tuned on TRAINING patients only; chb05 and Siena stay untouched)
# Seizures last many consecutive 2-s windows, so averaging neighbouring window probabilities removes isolated false alarms.
# Run after sections 1-13 (needs X_train, y_train, X_test, y_test, X_sp, y_s, m1, m5, file_ids, tr, te, ev, RF, XGBClassifier, ratio).
import numpy as np, pandas as pd
from scipy.ndimage import uniform_filter1d

def smooth(p, groups, k):
    out = p.copy()
    for g in pd.unique(groups):
        idx = np.where(groups == g)[0]
        out[idx] = uniform_filter1d(p[idx], size=k, mode='nearest') if k > 1 else p[idx]
    return out

pid_all = np.array([f[:5] for f in file_ids])
pid_tr, grp_tr, grp_te = pid_all[tr], np.array(file_ids)[tr], np.array(file_ids)[te]
KS  = [1, 3, 5, 7, 9, 15]
THS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60]

def make(name):
    if name == 'Baseline RF': return RF()
    return XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, scale_pos_weight=ratio, subsample=0.8,
                         colsample_bytree=0.8, eval_metric='logloss', random_state=42, verbosity=0)

rows = []
for name, final in [('Baseline RF', m1), ('XGBoost', m5)]:
    # inner patient-wise validation on the two TRAINING patients (train on one, validate on the other)
    folds = []
    for a, b in [('chb01', 'chb03'), ('chb03', 'chb01')]:
        ia, ib = pid_tr == a, pid_tr == b
        mm = make(name).fit(X_train[ia], y_train[ia])
        folds.append((mm.predict_proba(X_train[ib])[:, 1], y_train[ib], grp_tr[ib]))
    best, bestf = None, -1
    for k in KS:
        for t in THS:
            f1s = []
            for p, yy, gg in folds:
                ps = smooth(p, gg, k)
                f1s.append(ev(name, yy, (ps >= t).astype(int), ps)['F1'])
            if np.mean(f1s) > bestf: bestf, best = np.mean(f1s), (k, t)
    k, t = best
    print(f'{name}: window k={k}, threshold={t} chosen on training patients (validation F1 {bestf:.3f})')
    p = final.predict_proba(X_test)[:, 1]
    r0 = ev(name + ' (default)', y_test, (p >= 0.5).astype(int), p)
    ps = smooth(p, grp_te, k); r1 = ev(name + ' (smoothed+tuned)', y_test, (ps >= t).astype(int), ps)
    q = final.predict_proba(X_sp)[:, 1]
    s0 = ev(name + ' Siena (default)', y_s, (q >= 0.5).astype(int), q)
    qs = smooth(q, np.zeros(len(q)), k); s1 = ev(name + ' Siena (smoothed+tuned)', y_s, (qs >= t).astype(int), qs)
    rows += [r0, r1, s0, s1]
print(pd.DataFrame(rows).to_string(index=False))
