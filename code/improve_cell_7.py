# ===== CELL 7: (A) alarm post-processing on the saved Cell-6 probabilities, (B) baseline-period normalisation =====
# Needs Cells 1 and 2 run first (and Cell 4 for the Siena part). Cell 6 must have finished (it saved its probabilities to Drive).
import numpy as np, pandas as pd, time, os, gc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from scipy.ndimage import uniform_filter1d
from scipy.stats import wilcoxon, ttest_rel
assert 'Xf' in globals(), 'Run cells 1 and 2 first.'
t00 = time.time()
NEG_RATIO, CTX, BASE_WIN = 40, 5, 900            # BASE_WIN = 900 windows = 30 min of seizure-free baseline recording
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
def ctx(Z, groups, k=CTX):
    out = np.empty_like(Z)
    for g in pd.unique(groups):
        ix = np.where(groups == g)[0]; out[ix] = uniform_filter1d(Z[ix], size=k, axis=0, mode='nearest')
    return out
def runs(a):                                       # start/end (exclusive) indices of the runs of ones
    d = np.diff(np.concatenate([[0], a, [0]])); return np.where(d == 1)[0], np.where(d == -1)[0]
def event_metrics(yt, yp, groups):
    n_ev = det = fa = 0
    for g in pd.unique(groups):
        ix = np.where(groups == g)[0]; t, p = yt[ix], yp[ix]
        s, e = runs(t); n_ev += len(s); det += sum(int(p[a:b].any()) for a, b in zip(s, e))
        s, e = runs(p); fa += sum(int(t[a:b].sum() == 0) for a, b in zip(s, e))
    return n_ev, det, fa, len(yt) * 2 / 3600
def postproc(p, groups, thr, m, gap):              # threshold -> drop alarms shorter than m windows -> merge alarms closer than gap windows
    yp = np.zeros(len(p), dtype=int)
    for g in pd.unique(groups):
        ix = np.where(groups == g)[0]; a = (p[ix] >= thr).astype(np.int8)
        if m > 1:
            s, e = runs(a)
            for x, z in zip(s, e):
                if z - x < m: a[x:z] = 0
        if gap > 0:
            s, e = runs(a)
            for i in range(len(s) - 1):
                if s[i + 1] - e[i] < gap: a[e[i]:s[i + 1]] = 1
        yp[ix] = a
    return yp
GRID = [(t, m, g) for t in (0.1, 0.2, 0.3, 0.4, 0.5) for m in (1, 2, 3, 5) for g in (0, 5, 10, 20)]
LAMBDA = 0.1                                       # score = event sensitivity - LAMBDA * false alarms per hour (pooled over the other patients)
def select_and_apply(P, label):
    """P: dict patient -> (y, fid, prob). Parameters for each patient are chosen on the OTHER patients' out-of-fold probabilities."""
    tab = {pt: {c: event_metrics(yy, postproc(pp, ff, *c), ff) for c in GRID} for pt, (yy, ff, pp) in P.items()}
    out, chosen = [], {}
    for pt in P:
        best, bs = None, -9
        for c in GRID:
            ne = sum(tab[q][c][0] for q in P if q != pt); de = sum(tab[q][c][1] for q in P if q != pt)
            fa = sum(tab[q][c][2] for q in P if q != pt); hh = sum(tab[q][c][3] for q in P if q != pt)
            sc = de / max(ne, 1) - LAMBDA * fa / hh
            if sc > bs: bs, best = sc, c
        chosen[pt] = best; ne, de, fa, hh = tab[pt][best]; out.append((pt, ne, de, fa, fa / hh, best))
    od = pd.DataFrame(out, columns=['patient', 'events', 'detected', 'false_alarms', 'FA_per_hour', 'params'])
    print(f'\n=== {label}: event-level with post-processing (params chosen on the other patients) ===')
    print(f'seizures detected {int(od.detected.sum())}/{int(od.events.sum())} ({100*od.detected.sum()/od.events.sum():.1f}%) | false alarms/hour: mean {od.FA_per_hour.mean():.2f}, median {od.FA_per_hour.median():.2f}')
    print('patients with >=1 detected:', int((od.detected > 0).sum()), '/', len(od), '| most frequent params (thr, min_len, gap):', pd.Series([str(c) for c in chosen.values()]).mode()[0])
    print(od.round(2).to_string(index=False)); return od, chosen

# ---------------- (A) post-processing of the stored Cell-6 probabilities ----------------
rows6 = pd.read_csv(f'{CACHE}/improved_rows.csv'); sm6 = rows6[rows6.Model.str.contains('smoothing')].set_index('patient')
P6 = {}
for pt in sm6.index:
    d = np.load(f'{CACHE}/improved_pp_{pt}.npz'); P6[pt] = (d['y'], d['fid'], d['ppD'])
ev0 = [event_metrics(yy, (pp >= sm6.loc[pt, 'thr']).astype(int), ff) for pt, (yy, ff, pp) in P6.items()]
print('=== Cell-6 pipeline as before (selected k and threshold, no post-processing) ===')
print(f'seizures detected {sum(e[1] for e in ev0)}/{sum(e[0] for e in ev0)} | false alarms/hour mean {np.mean([e[2]/e[3] for e in ev0]):.2f}')
odA, chosenA = select_and_apply(P6, 'Cell-6 pipeline + post-processing')

# ---------------- (B) baseline-period normalisation (no seizure windows in the statistics) ----------------
def base_norm(X, groups, yy, ff):
    out = np.empty(X.shape, dtype=np.float32)
    for g in pd.unique(groups):
        ix = np.where(groups == g)[0]; A = np.sign(X[ix]) * np.log1p(np.abs(X[ix]))
        files_ns = [f for f in pd.unique(ff[ix]) if yy[ix][ff[ix] == f].sum() == 0]          # seizure-free recordings of this patient
        b = np.concatenate([np.where(ff[ix] == f)[0] for f in files_ns])[:BASE_WIN] if files_ns else np.array([], int)
        if len(b) < 300: b = np.where(yy[ix] == 0)[0][:BASE_WIN]                              # fallback: first seizure-free windows
        med = np.median(A[b], 0); q1, q3 = np.percentile(A[b], [25, 75], axis=0); iqr = q3 - q1
        out[ix] = (A - med) / np.where(iqr > 1e-6, iqr, A[b].std(0) + 1e-6)
    return np.clip(out, -20, 20)
Xb = base_norm(Xf, pid, y, fid); print(f'\nbaseline-period normalisation done [{time.time()-t00:.0f}s]')
test_pts = [p for p in PATIENTS if y[pid == p].sum() > 0]
rb_csv = f'{CACHE}/improved_b_rows.csv'
rowsB = pd.read_csv(rb_csv).to_dict('records') if os.path.exists(rb_csv) else []
doneB = {r['patient'] for r in rowsB}
for pt in test_pts:
    if pt in doneB: print(pt, 'already done'); continue
    t0 = time.time(); tr, te = pid != pt, pid == pt; tri = sub(tr, 0)
    sc = StandardScaler().fit(Xb[tri]); pca = PCA(n_components=120, random_state=42).fit(sc.transform(Xb[tri]))
    Z = pca.transform(sc.transform(Xb)).astype(np.float32); Zc = np.hstack([Z, ctx(Z, fid)])
    yte, fte = y[te], fid[te]; ppC = RF().fit(Zc[tri], y[tri]).predict_proba(Zc[te])[:, 1]
    cand = [p for p in test_pts if p != pt]; rng = np.random.RandomState(0); inner = []
    for q in rng.choice(cand, size=min(4, len(cand)), replace=False):
        itr = sub((pid != pt) & (pid != q), 1); iva = pid == q
        mm = RF(20).fit(Zc[itr], y[itr]); inner.append((mm.predict_proba(Zc[iva])[:, 1], y[iva], fid[iva]))
    best, bf = (1, 0.5), -1
    for k in KS:
        for t in THS:
            f1s = [ev('', yy, (smooth(p_, g_, k) >= t).astype(int), smooth(p_, g_, k))['F1'] for p_, yy, g_ in inner]
            if np.mean(f1s) > bf: bf, best = np.mean(f1s), (k, t)
    k, t = best; ppD = smooth(ppC, fte, k)
    r = ev('RF + baseline-norm + context + smoothing', yte, (ppD >= t).astype(int), ppD); r.update(patient=pt, k=k, thr=t); rowsB.append(r)
    np.savez_compressed(f'{CACHE}/improved_b_pp_{pt}.npz', y=yte, fid=fte, ppD=ppD)
    pd.DataFrame(rowsB).to_csv(rb_csv, index=False)
    print(f'{pt}: AUC {r["AUC"]:.3f} F1 {r["F1"]:.3f} (k={k}, thr={t}) [{time.time()-t0:.0f}s]')
    del Z, Zc; gc.collect()
dfB = pd.DataFrame(rowsB).set_index('patient')
print('\n=== Window-level, baseline-period normalisation (mean +- std over patients) ===')
print({kx: f'{dfB[kx].mean():.3f} +- {dfB[kx].std(ddof=0):.3f}' for kx in ['P', 'R', 'F1', 'AUC', 'FDR']})
print('=== Cell-6 full pipeline (whole-recording normalisation) for comparison ===')
s6 = sm6.loc[dfB.index]; print({kx: f'{s6[kx].mean():.3f} +- {s6[kx].std(ddof=0):.3f}' for kx in ['P', 'R', 'F1', 'AUC', 'FDR']})
old_f = f'{CACHE}/lopo_results.csv'
if os.path.exists(old_f):
    base = pd.read_csv(old_f); base = base[base.Model == 'Baseline RF'].set_index('patient').loc[dfB.index]
    print('\n=== Paired tests: baseline-period pipeline vs earlier raw-feature Baseline RF ===')
    for kx in ['F1', 'AUC', 'FDR']:
        try: w = wilcoxon(base[kx], dfB[kx]).pvalue
        except ValueError: w = float('nan')
        print(f'{kx:3s} t-test p={ttest_rel(base[kx], dfB[kx]).pvalue:.4f}  Wilcoxon p={w:.4f}  better in {(dfB[kx] > base[kx]).sum() if kx != "FDR" else (dfB[kx] < base[kx]).sum()}/{len(dfB)}')
PB = {}
for pt in dfB.index:
    d = np.load(f'{CACHE}/improved_b_pp_{pt}.npz'); PB[pt] = (d['y'], d['fid'], d['ppD'])
evb = [event_metrics(yy, (pp >= dfB.loc[pt, 'thr']).astype(int), ff) for pt, (yy, ff, pp) in PB.items()]
print(f'\nEvent-level without post-processing: seizures detected {sum(e[1] for e in evb)}/{sum(e[0] for e in evb)} | false alarms/hour mean {np.mean([e[2]/e[3] for e in evb]):.2f}')
odB, chosenB = select_and_apply(PB, 'Baseline-period pipeline + post-processing')

# ---------------- Siena with baseline-period normalisation ----------------
if os.path.exists(f'{CACHE}/siena.npz'):
    d = np.load(f'{CACHE}/siena.npz'); XsS = np.nan_to_num(d['X'], nan=0.0, posinf=0.0, neginf=0.0); ysS, fsS, psS = d['y'], d['fid'], d['pid']
    XsB = base_norm(XsS, psS, ysS, fsS); ai = sub(np.ones(len(y), bool), 0)
    sc = StandardScaler().fit(Xb[ai]); pca = PCA(n_components=120, random_state=42).fit(sc.transform(Xb[ai]))
    Zall = pca.transform(sc.transform(Xb)).astype(np.float32); Zca = np.hstack([Zall, ctx(Zall, fid)])
    Zs = pca.transform(sc.transform(XsB)).astype(np.float32); Zsc = np.hstack([Zs, ctx(Zs, fsS)])
    kk = int(dfB.k.mode()[0]); tt = float(dfB.thr.median())
    pp = smooth(RF(100).fit(Zca[ai], y[ai]).predict_proba(Zsc)[:, 1], fsS, kk)
    pc = pd.Series([str(c) for c in chosenB.values()]).mode()[0]; thr_, m_, g_ = eval(pc)
    r1 = ev(f'baseline-norm + context + smoothing (k={kk}, thr={tt})', ysS, (pp >= tt).astype(int), pp)
    ne, de, fa, hh = event_metrics(ysS, (pp >= tt).astype(int), fsS)
    ne2, de2, fa2, hh2 = event_metrics(ysS, postproc(pp, fsS, thr_, m_, g_), fsS)
    print(f'\n=== CHB-MIT (22) -> Siena, baseline-period normalisation ({len(set(psS))} subjects, {int(ysS.sum())} seizure windows) ===')
    print(pd.DataFrame([r1]).round(3).to_string(index=False))
    print(f'  events: detected {de}/{ne}, false alarms/hour {fa/hh:.2f}')
    print(f'  with post-processing {pc}: detected {de2}/{ne2}, false alarms/hour {fa2/hh2:.2f}')
else:
    print('siena.npz not found: run Cell 4 first to include the Siena test.')
print('\nSend me ALL of this output.  Total time %.0f min' % ((time.time() - t00) / 60))
