# ===== CELL 6: IMPROVED PIPELINE (per-patient normalisation + temporal context + event-level metrics) =====
# Needs cells 1 and 2 run first (they only reload the cached features: Xf, y, pid, fid, PATIENTS, CACHE).
import numpy as np, pandas as pd, time, os, gc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from scipy.ndimage import uniform_filter1d
from scipy.stats import wilcoxon, ttest_rel
assert 'Xf' in globals(), 'Run cells 1 and 2 first (they only reload cached features).'

NEG_RATIO = 40
CTX = 5                                             # windows averaged for the context features (10 s)
KS, THS = [1, 3, 5, 7, 9, 15], [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60]
def RF(n=50, seed=42): return RandomForestClassifier(n_estimators=n, max_depth=10, random_state=seed, n_jobs=-1)
def sub(mask, seed):
    idx = np.where(mask)[0]; pos = idx[y[idx] == 1]; neg = idx[y[idx] == 0]
    neg = np.random.RandomState(seed).choice(neg, size=min(len(neg), NEG_RATIO * max(len(pos), 1)), replace=False)
    return np.sort(np.concatenate([pos, neg]))
def ev(name, yt, yp, pp):
    n = len(yt); fp = int(((yp == 1) & (yt == 0)).sum())
    return {'Model': name, 'P': precision_score(yt, yp, zero_division=0), 'R': recall_score(yt, yp, zero_division=0),
            'F1': f1_score(yt, yp, zero_division=0), 'AUC': roc_auc_score(yt, pp), 'FDR': fp / (n * 2 / 3600)}
def smooth(p, groups, k):
    if k <= 1: return p
    out = p.copy()
    for g in pd.unique(groups):
        ix = np.where(groups == g)[0]; out[ix] = uniform_filter1d(p[ix], size=k, mode='nearest')
    return out
def ctx(Z, groups, k=CTX):                          # moving average of the PCA features inside each file
    out = np.empty_like(Z)
    for g in pd.unique(groups):
        ix = np.where(groups == g)[0]; out[ix] = uniform_filter1d(Z[ix], size=k, axis=0, mode='nearest')
    return out
def patient_norm(X, groups):                        # signed log, then robust centring/scaling with the statistics of each patient (no labels used)
    out = np.empty(X.shape, dtype=np.float32)
    for g in pd.unique(groups):
        ix = np.where(groups == g)[0]; A = np.sign(X[ix]) * np.log1p(np.abs(X[ix]))
        med = np.median(A, 0); q1, q3 = np.percentile(A, [25, 75], axis=0); iqr = q3 - q1
        out[ix] = (A - med) / np.where(iqr > 1e-6, iqr, A.std(0) + 1e-6)
    return np.clip(out, -20, 20)
def event_metrics(yt, yp, groups):
    n_ev = det = fa = 0
    for g in pd.unique(groups):
        ix = np.where(groups == g)[0]; t, p = yt[ix], yp[ix]
        for arr, is_true in ((t, True), (p, False)):
            d = np.diff(np.concatenate([[0], arr, [0]])); st, en = np.where(d == 1)[0], np.where(d == -1)[0]
            for s, e in zip(st, en):
                if is_true: n_ev += 1; det += int(p[s:e].any())
                else: fa += int(t[s:e].sum() == 0)
    return n_ev, det, fa, fa / (len(yt) * 2 / 3600)

t00 = time.time()
Xn = patient_norm(Xf, pid); print(f'per-patient normalisation done [{time.time()-t00:.0f}s]')
test_pts = [p for p in PATIENTS if y[pid == p].sum() > 0]
rf_csv, ev_csv = f'{CACHE}/improved_rows.csv', f'{CACHE}/improved_events.csv'
rows = pd.read_csv(rf_csv).to_dict('records') if os.path.exists(rf_csv) else []
erows = pd.read_csv(ev_csv).to_dict('records') if os.path.exists(ev_csv) else []
done = {r['patient'] for r in rows}
for pt in test_pts:
    if pt in done: print(pt, 'already done'); continue
    t0 = time.time(); tr, te = pid != pt, pid == pt; tri = sub(tr, 0)
    sc = StandardScaler().fit(Xn[tri]); pca = PCA(n_components=120, random_state=42).fit(sc.transform(Xn[tri]))
    Z = pca.transform(sc.transform(Xn)).astype(np.float32); Zc = np.hstack([Z, ctx(Z, fid)])
    yte, fte = y[te], fid[te]
    mB, mC = RF().fit(Z[tri], y[tri]), RF().fit(Zc[tri], y[tri])
    ppB, ppC = mB.predict_proba(Z[te])[:, 1], mC.predict_proba(Zc[te])[:, 1]
    cand = [p for p in test_pts if p != pt]; rng = np.random.RandomState(0)
    inner = []
    for q in rng.choice(cand, size=min(4, len(cand)), replace=False):      # smoothing parameters chosen on training patients only
        itr = sub((pid != pt) & (pid != q), 1); iva = pid == q
        mm = RF(20).fit(Zc[itr], y[itr]); inner.append((mm.predict_proba(Zc[iva])[:, 1], y[iva], fid[iva]))
    best, bf = (1, 0.5), -1
    for k in KS:
        for t in THS:
            f1s = [ev('', yy, (smooth(p_, g_, k) >= t).astype(int), smooth(p_, g_, k))['F1'] for p_, yy, g_ in inner]
            if np.mean(f1s) > bf: bf, best = np.mean(f1s), (k, t)
    k, t = best; ppD = smooth(ppC, fte, k)
    for name, pp, thr in (('RF + patient norm', ppB, 0.5), ('RF + norm + context', ppC, 0.5), ('RF + norm + context + smoothing', ppD, t)):
        yp = (pp >= thr).astype(int); r = ev(name, yte, yp, pp); r['patient'] = pt; r['k'] = k if 'smoothing' in name else 1; r['thr'] = thr; rows.append(r)
        ne, de, fa, fah = event_metrics(yte, yp, fte); erows.append({'patient': pt, 'Model': name, 'events': ne, 'detected': de, 'false_alarms': fa, 'FA_per_hour': fah})
    np.savez_compressed(f'{CACHE}/improved_pp_{pt}.npz', y=yte, fid=fte, ppB=ppB, ppC=ppC, ppD=ppD)
    pd.DataFrame(rows).to_csv(rf_csv, index=False); pd.DataFrame(erows).to_csv(ev_csv, index=False)
    print(f'{pt}: AUC norm {rows[-3]["AUC"]:.3f} | +ctx {rows[-2]["AUC"]:.3f} | +smooth {rows[-1]["AUC"]:.3f} (k={k}, thr={t}) [{time.time()-t0:.0f}s]')
    del Z, Zc; gc.collect()

df = pd.DataFrame(rows); ed = pd.DataFrame(erows)
print('\n=== Window-level: mean +- std over held-out patients ===')
agg = [{'Model': n, 'n': len(g), **{kx: f'{g[kx].mean():.3f} +- {g[kx].std(ddof=0):.3f}' for kx in ['P', 'R', 'F1', 'AUC', 'FDR']}} for n, g in df.groupby('Model', sort=False)]
print(pd.DataFrame(agg).to_string(index=False))
old_f = f'{CACHE}/lopo_results.csv'
if os.path.exists(old_f):
    old = pd.read_csv(old_f); base = old[old.Model == 'Baseline RF'].set_index('patient'); sm = old[old.Model == 'Baseline RF + smoothing'].set_index('patient')
    print('\n=== Earlier results for comparison (same patients) ===')
    for nm, o in (('Baseline RF (raw features)', base), ('Baseline RF + smoothing (raw features)', sm)):
        o = o.loc[[p for p in base.index if p in set(df.patient)]]; print(nm, {kx: f'{o[kx].mean():.3f} +- {o[kx].std(ddof=0):.3f}' for kx in ['P', 'R', 'F1', 'AUC', 'FDR']})
    print('\n=== Paired tests: improved models vs earlier Baseline RF ===')
    for name in df.Model.unique():
        o = df[df.Model == name].set_index('patient'); b = base.loc[o.index]
        for kx in ['F1', 'AUC', 'FDR']:
            try: w = wilcoxon(b[kx], o[kx]).pvalue
            except ValueError: w = float('nan')
            print(f'{kx:3s} {name:34s} t-test p={ttest_rel(b[kx], o[kx]).pvalue:.4f}  Wilcoxon p={w:.4f}  better in {(o[kx] > b[kx]).sum() if kx != "FDR" else (o[kx] < b[kx]).sum()}/{len(o)}')
print('\n=== Event-level (all held-out patients pooled) ===')
for n, g in ed.groupby('Model', sort=False):
    print(f'{n:34s} seizures detected {int(g.detected.sum())}/{int(g.events.sum())} ({100*g.detected.sum()/max(g.events.sum(),1):.1f}%) | false alarms/hour: mean {g.FA_per_hour.mean():.2f}, median {g.FA_per_hour.median():.2f}')
print('\nPer-patient event table:'); print(ed.round(2).to_string(index=False))

# ---------------- same models -> Siena, with per-subject normalisation ----------------
if os.path.exists(f'{CACHE}/siena.npz'):
    d = np.load(f'{CACHE}/siena.npz'); XsS = np.nan_to_num(d['X'], nan=0.0, posinf=0.0, neginf=0.0); ysS, fsS, psS = d['y'], d['fid'], d['pid']
    XsN = patient_norm(XsS, psS); ai = sub(np.ones(len(y), bool), 0)
    sc = StandardScaler().fit(Xn[ai]); pca = PCA(n_components=120, random_state=42).fit(sc.transform(Xn[ai]))
    Zall = pca.transform(sc.transform(Xn)).astype(np.float32); Zca = np.hstack([Zall, ctx(Zall, fid)])
    Zs = pca.transform(sc.transform(XsN)).astype(np.float32); Zsc = np.hstack([Zs, ctx(Zs, fsS)])
    kk = int(df[df.Model.str.contains('smoothing')].k.mode()[0]); tt = float(df[df.Model.str.contains('smoothing')].thr.median())
    mB, mC = RF(100).fit(Zall[ai], y[ai]), RF(100).fit(Zca[ai], y[ai])
    ppB, ppC = mB.predict_proba(Zs)[:, 1], mC.predict_proba(Zsc)[:, 1]; ppD = smooth(ppC, fsS, kk)
    out, eo = [], []
    for name, pp, thr in (('RF + subject norm', ppB, 0.5), ('RF + norm + context', ppC, 0.5), (f'RF + norm + context + smoothing (k={kk}, thr={tt})', ppD, tt)):
        yp = (pp >= thr).astype(int); out.append(ev(name, ysS, yp, pp)); ne, de, fa, fah = event_metrics(ysS, yp, fsS); eo.append((de, ne, fah))
    print(f'\n=== CHB-MIT (22 patients) -> Siena, per-subject normalisation ({len(set(psS))} subjects, {len(ysS)} windows, {int(ysS.sum())} seizure windows) ===')
    print(pd.DataFrame(out).round(3).to_string(index=False))
    for r, (de, ne, fah) in zip(out, eo): print(f'  {r["Model"]}: seizures detected {de}/{ne}, false alarms/hour {fah:.2f}')
else:
    print('siena.npz not found: run cell 4 first to include the Siena test.')
print('\nSend me ALL of this output.  Total time %.0f min' % ((time.time() - t00) / 60))
