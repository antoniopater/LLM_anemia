import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier

# 1. Symulacja danych „realnych”
def generate_group_data(label, n_samples=30000):
    params = {
        "Anemia Mikrocytarna": {'RBC':(4.0,0.5),'HGB':(11.0,1.0),'HCT':(33.0,2.0),
                                 'MCV':(70.0,3.0),'MCH':(23.0,2.0),'MCHC':(32.0,1.0),
                                 'RDW':(15.0,1.5),'PLT':(300,50),'WBC':(7.0,1.0)},
        "Anemia Makrocytarna": {'RBC':(3.8,0.5),'HGB':(10.5,1.0),'HCT':(31.0,2.0),
                                 'MCV':(105.0,10.0),'MCH':(35.0,3.0),'MCHC':(33.0,1.0),
                                 'RDW':(16.0,1.5),'PLT':(200,30),'WBC':(7.0,1.0)},
        "Anemia Normocytarna": {'RBC':(4.5,0.4),'HGB':(12.5,1.2),'HCT':(37.0,2.0),
                                 'MCV':(90.0,5.0),'MCH':(30.0,2.0),'MCHC':(33.0,1.0),
                                 'RDW':(14.0,1.0),'PLT':(250,40),'WBC':(8.0,1.0)},
        "Anemia Hemolityczna":{'RBC':(3.2,0.4),'HGB':(9.5,1.0),'HCT':(28.0,2.5),
                                 'MCV':(88.0,4.0),'MCH':(29.5,1.5),'MCHC':(33.0,1.0),
                                 'RDW':(17.0,2.0),'PLT':(320,60),'WBC':(10.5,1.5)},
        "Anemia Aplastyczna": {'RBC':(2.8,0.4),'HGB':(8.5,1.0),'HCT':(26.0,2.0),
                                 'MCV':(92.0,4.0),'MCH':(30.0,1.5),'MCHC':(33.0,0.8),
                                 'RDW':(14.5,1.5),'PLT':(60,20),'WBC':(2.5,0.8)},
        "Healthy":            {'RBC':(5.0,0.5),'HGB':(16.0,1.2),'HCT':(42.0,2.0),
                                 'MCV':(90.0,5.0),'MCH':(29.0,2.0),'MCHC':(34.0,1.0),
                                 'RDW':(13.5,1.0),'PLT':(280,40),'WBC':(7.0,1.0)}
    }
    df = pd.DataFrame({k: np.random.normal(loc, scale, n_samples)
                       for k,(loc,scale) in params[label].items()})
    df['Label'] = label
    return df

#  Lista klas
groups = [
    "Anemia Mikrocytarna","Anemia Makrocytarna","Anemia Normocytarna",
    "Anemia Hemolityczna","Anemia Aplastyczna","Healthy"
]

# Generacja i wymieszanie realnych danych
df_real = pd.concat([generate_group_data(g) for g in groups], ignore_index=True)
df_real = df_real.sample(frac=1, random_state=42).reset_index(drop=True)

# 2. Przygotowanie tensorów cech + one-hot labels
feature_cols = ['RBC','HGB','HCT','MCV','MCH','MCHC','RDW','PLT','WBC']
label_ohe    = pd.get_dummies(df_real['Label'])[groups]
df_all       = pd.concat([df_real[feature_cols], label_ohe], axis=1).astype('float32')

# **tu poprawka**: jawne rzutowanie na float32
X = torch.from_numpy(df_all.values.astype(np.float32))
dataset = TensorDataset(X)
loader  = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)

# 3. Definicja CVAE
feature_dim = len(feature_cols)
label_dim   = len(groups)
input_dim   = feature_dim + label_dim
latent_dim  = 10

class CVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim,128), nn.ReLU(),
            nn.Linear(128,64), nn.ReLU()
        )
        self.mu_layer     = nn.Linear(64, latent_dim)
        self.logvar_layer = nn.Linear(64, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim+label_dim,64), nn.ReLU(),
            nn.Linear(64,128), nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    def encode(self, x):
        h = self.enc(x)
        return self.mu_layer(h), self.logvar_layer(h)
    def reparam(self, mu, logvar):
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)
    def decode(self, z, lbl):
        return self.dec(torch.cat([z, lbl], dim=1))
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        lbl = x[:, feature_dim:]
        return self.decode(z, lbl), mu, logvar

def loss_fn(recon, x, mu, logvar, beta=1.0):
    mse = nn.MSELoss(reduction='sum')(recon[:,:feature_dim], x[:,:feature_dim])
    bce = nn.BCEWithLogitsLoss(reduction='sum')(recon[:,feature_dim:], x[:,feature_dim:])
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + bce + beta * kld

# 4. Trening CVAE
device = torch.device('cpu')
cvae   = CVAE().to(device)
opt    = optim.Adam(cvae.parameters(), lr=1e-3)
for ep in range(1, 51):
    cvae.train(); total=0
    for (batch,) in loader:
        x = batch.to(device)
        opt.zero_grad()
        recon, mu, logvar = cvae(x)
        loss = loss_fn(recon, x, mu, logvar)
        loss.backward(); opt.step()
        total += loss.item()
    print(f"Epoch {ep}, Loss {total/len(dataset):.4f}")

# 5. Generacja syntetycznych danych z soft-labels
cvae.eval()
with torch.no_grad():
    n    = len(df_real)
    z    = torch.randn(n, latent_dim)
    lbls = torch.zeros(n, label_dim)
    for i in range(label_dim):
        lbls[i*(n//label_dim):(i+1)*(n//label_dim), i] = 1
    out   = cvae.decode(z, lbls)
    feats = out[:, :feature_dim].numpy()
    probs = torch.sigmoid(out[:, feature_dim:]).numpy()

    df_syn = pd.DataFrame(feats, columns=feature_cols)
    for i, g in enumerate(groups):
        df_syn[f'Prob_{g}'] = probs[:, i]
    df_syn['Label_argmax'] = [groups[i] for i in probs.argmax(axis=1)]
    df_syn.to_csv("synthetic_data_with_probs.csv", index=False)
    print("Zapisano synthetic_data_with_probs.csv")

# poniżej ewaluacja XGBoost, ROC, itp. – pozostawiasz bez zmian.
