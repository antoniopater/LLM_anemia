import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.express as px

pca = PCA(n_components=3)

scaler = StandardScaler()

df = pd.read_csv("synthetic_data_vae.csv")


label_cols = ['Label_Anemia Makrocytarna', 'Label_Anemia Mikrocytarna', 'Label_Anemia Normocytarna']

df['Label'] = df[label_cols].idxmax(axis=1)

df['Label'] = df['Label'].str.replace('Label_', '')

label_mapping = {
    "Anemia Mikrocytarna": 0,
    "Anemia Makrocytarna": 1,
    "Anemia Normocytarna": 2,
    "Healthy":4
}
df['Label_num'] = df['Label'].map(label_mapping)

features = ['RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'RDW', 'PLT', 'WBC']
X = df[features]
y = df['Label_num']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Create DataFrame for Plotly
df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2", "PC3"])
df_pca['Label_num'] = y

# Create 3D scatter plot using Plotly
fig = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3', color='Label_num',
                    labels={"Label_num": "Klasa choroby"},
                    title="Wizualizacja PCA (3D)",
                    color_continuous_scale='Viridis')

# Show the interactive plot
fig.show()