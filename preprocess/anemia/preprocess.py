import os
import joblib
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = None
pca = None


def scaleData(X, n_components=3):
    global scaler, pca
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    scaler_path = os.path.join(script_dir, "scaler.pkl")
    pca_path = os.path.join(script_dir, "pca.pkl")

    # Save the scaler and PCA objects
    joblib.dump(scaler, scaler_path)
    joblib.dump(pca, pca_path)

    return X_pca


def changeLabels():
    # df = pd.read_csv("../../trainingData/anemia/synthetic_data_vae3.csv")
    df = pd.read_csv("../../trainingData/anemia/synthetic_data_vae3.csv")
    label_cols = ['Label_Anemia Makrocytarna',
                  'Label_Anemia Mikrocytarna',
                  'Label_Anemia Normocytarna',
                  'Label_Anemia Hemolityczna',
                  'Label_Anemia Aplastyczna',
                  'Label_Trombocytopenia',
                  'Label_Healthy'
                  ]
    missing_cols = [col for col in label_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Brakuje kolumn w pliku CSV: {missing_cols}")

    df['Label'] = df[label_cols].idxmax(axis=1)
    df['Label'] = df['Label'].str.replace('Label_', '')

    label_mapping = {
        "Anemia Mikrocytarna": 0,
        "Anemia Makrocytarna": 1,
        "Anemia Hemolityczna": 2,
        "Anemia Aplastyczna": 3,
        "Trombocytopenia": 4,
        "Healthy": 5
    }
    df['Label_num'] = df['Label'].map(label_mapping)

    return df


if __name__ == "__main__":
    df = changeLabels()

    print(df['Label'].value_counts())

    print(df.shape)
