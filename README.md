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

- `paper/` – camera-ready version (PDF and DOCX).
- `code/seizure_MULTI_PATIENT_22.ipynb` – leave-one-patient-out evaluation over 22 CHB-MIT patients (Baseline RF, SMOTE+RF, Balanced RF, XGBoost) and CHB-MIT → Siena test.
- `code/improve_cell_6.py` – per-patient normalisation, context features, temporal smoothing, event-level evaluation, alarm post-processing.
- `code/improve_cell_7.py` – baseline-period normalisation variant (30-minute seizure-free baseline).

Data: CHB-MIT and Siena Scalp EEG databases from PhysioNet (not included).
