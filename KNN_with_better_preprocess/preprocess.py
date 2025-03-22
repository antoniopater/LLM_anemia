import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
scaler = None
pca = None
def fit_scale_pca(X, n_components=3):
    global scaler,pca
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca
def transform_scale_pca(X):
    global scaler,pca
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    return X_pca
def chngeLabels():
    df = pd.read_csv("synthetic_data_vae.csv")

    label_cols = ['Label_Anemia Makrocytarna', 'Label_Anemia Mikrocytarna',
                  'Label_Anemia Normocytarna', 'Label_Healthy']

    df['Label'] = df[label_cols].idxmax(axis=1)
    df['Label'] = df['Label'].str.replace('Label_', '')

    label_mapping = {
        "Anemia Mikrocytarna": 0,
        "Anemia Makrocytarna": 1,
        "Anemia Normocytarna": 2,
        "Healthy": 3
    }
    df['Label_num'] = df['Label'].map(label_mapping)

    return df