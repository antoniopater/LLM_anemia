import pandas as pd
import plotly.express as px
import preprocess.anemia.preprocess as preprocessData

df = preprocessData.changeLabels()

features = ['RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'RDW', 'PLT', 'WBC']
X = df[features]
y = df['Label_num']

# Scaler

X_pca = preprocessData.scaleData(X)

df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2", "PC3"])
df_pca['Label_num'] = y


fig = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3', color='Label_num',
                    labels={"Label_num": "Klasa choroby"},
                    title="Wizualizacja PCA (3D)",
                    color_continuous_scale='Viridis')


fig.show()
