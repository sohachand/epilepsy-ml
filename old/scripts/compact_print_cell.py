# 14. Compact re-print (the PDF cut off the right-hand columns). Run AFTER sections 12b, 13a, 13b. Send me the output.
import numpy as np
print('--- 12b: 5 seeds, mean +- std (CHB-MIT chb05)')
for n, rs in seed_res.items():
    print(n, '|', ' '.join(f"{k}={np.mean([r[k] for r in rs]):.3f}+-{np.std([r[k] for r in rs]):.3f}" for k in ['P','R','F1','AUC','FDR']))
print('--- 13a: LOPO mean +- std over 3 patients')
for n, g in df_lopo.groupby('Model', sort=False):
    print(n, '|', ' '.join(f"{k}={g[k].mean():.3f}+-{g[k].std(ddof=0):.3f}" for k in ['P','R','F1','AUC','FDR']))
print('--- 13b: deep baselines, 5 seeds')
for arch in (CNN1D, MLP):
    rc, rs_ = [], []
    for s in [42, 123, 456, 789, 999]:
        m = train_cnn(s, arch)
        pc = predict_cnn(m, X_test); ps = predict_cnn(m, X_sp)
        rc.append(ev(arch.__name__, y_test, (pc >= 0.5).astype(int), pc))
        rs_.append(ev(arch.__name__, y_s, (ps >= 0.5).astype(int), ps))
    for tag, res in (('chb05', rc), ('Siena', rs_)):
        print(arch.__name__, tag, '|', ' '.join(f"{k}={np.mean([r[k] for r in res]):.3f}+-{np.std([r[k] for r in res]):.3f}" for k in ['P','R','F1','AUC','FDR']))
