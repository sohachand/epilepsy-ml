Epilepsy Detection using Machine Learning
Overview

This project focuses on detecting epileptic seizures using EEG data and machine learning techniques. The goal is to identify patterns in brain signals that indicate seizure activity.

Features
EEG data preprocessing
Feature extraction
Model training (Random Forest / XGBoost / etc.)
Performance evaluation
Tech Stack
Python
Scikit-learn
NumPy, Pandas
Matplotlib / Seaborn
Results

The model achieves accurate classification of seizure vs non-seizure states based on extracted features.

How to Run
Open the notebook in Google Colab
Upload dataset
Run all cells

Dataset

https://physionet.org/content/chbmit/1.0.0/

Author

Soha Chand

## IEEE MINDS 2026 paper (Paper ID 958)

"Epileptic Seizure Detection Using Machine Learning: A Comparative Study of Class Imbalance Handling Techniques" – Soha Chand, Atharv Dixit, Dr. Hima Deepthi Vankayalapati (MPSTME, SVKM's NMIMS).

### Repository layout

| Folder | Contents |
|---|---|
| `paper/final/` | **Final** camera-ready paper (v10, PDF + DOCX) and the response to reviewers |
| `new_final_code/` | **Current** code behind the final paper: `seizure_MULTI_PATIENT_22.ipynb` (22-patient leave-one-patient-out on CHB-MIT, CHB-MIT to Siena test), `improve_cell_6.py` (per-patient normalisation, context, smoothing, event-level evaluation, alarm post-processing), `improve_cell_7.py` (baseline-period normalisation variant) |
| `old/notebooks/` | **Old** notebooks: earlier two-patient study versions (chb01, chb03 train; chb05 test) and intermediate experiments |
| `old/scripts/` | **Old** helper scripts |
| `old/paper_versions/` | **Old** paper drafts and earlier camera-ready versions (v2 to v9) |
| `old/figures/` | **Old** figure images |

Note: the two-patient results in the paper (seed analysis, threshold study, PCA ablation, deep-learning baselines) come from the notebooks in `old/notebooks/`.

Data: CHB-MIT and Siena Scalp EEG databases from PhysioNet (not included).
