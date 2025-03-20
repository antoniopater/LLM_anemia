import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def scaleData(X):
    pca = PCA(n_components=3)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca

def changeLabels() -> pd.DataFrame:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    data_folder = os.path.join(project_root, 'data')
    file_path = os.path.join(data_folder, 'synthetic_data_vae.csv')

    df = pd.read_csv(file_path)

    label_cols = ['Label_Anemia Makrocytarna', 'Label_Anemia Mikrocytarna', 'Label_Anemia Normocytarna',
                  'Label_Healthy']

    df['Label'] = df[label_cols].idxmax(axis=1)

    # Usuń prefiks 'Label_' z nazwy etykiety
    df['Label'] = df['Label'].str.replace('Label_', '')

    # Mapowanie etykiet na wartości numeryczne
    label_mapping = {
        "Anemia Mikrocytarna": 0,
        "Anemia Makrocytarna": 1,
        "Anemia Normocytarna": 2,
        "Healthy": 3
    }
    df['Label_num'] = df['Label'].map(label_mapping)

    return df
