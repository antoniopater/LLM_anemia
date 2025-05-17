import numpy as np
import pandas as pd
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt

# --- Ustawienia ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
batch_size = 128
epochs = 100
learning_rate = 1e-3
latent_dim = 10

# --- Funkcja symulująca dane ---
def generate_group_data(label, n_samples=random.randint(27000, 33000)):
    if label == "Anemia Mikrocytarna":
        params = dict(RBC=(4.0,0.5), HGB=(11.0,1.0), HCT=(33.0,2.0),
                      MCV=(70.0,5.0), MCH=(23.0,2.0), MCHC=(32.0,1.0),
                      RDW=(15.0,1.5), PLT=(300,50), WBC=(7.0,1.0))
    elif label == "Anemia Makrocytarna":
        params = dict(RBC=(3.8,0.5), HGB=(10.5,1.0), HCT=(31.0,2.0),
                      MCV=(105.0,10.0), MCH=(35.0,3.0), MCHC=(33.0,1.0),
                      RDW=(16.0,1.5), PLT=(200,30), WBC=(7.0,1.0))
    elif label == "Anemia Normocytarna":
        params = dict(RBC=(4.0,0.5), HGB=(11.0,1.0), HCT=(33.0,2.0),
                      MCV=(90.0,5.0), MCH=(30.0,2.0), MCHC=(33.0,1.0),
                      RDW=(15.0,1.5), PLT=(250,40), WBC=(8.0,1.0))
    elif label == "Anemia Hemolityczna":
        params = dict(RBC=(3.2,0.4), HGB=(9.5,1.0), HCT=(28.0,2.5),
                      MCV=(88.0,4.0), MCH=(29.5,1.5), MCHC=(33.0,1.0),
                      RDW=(17.0,2.0), PLT=(320,60), WBC=(10.5,1.5))
    elif label == "Anemia Aplastyczna":
        params = dict(RBC=(2.8,0.4), HGB=(8.5,1.0), HCT=(26.0,2.0),
                      MCV=(92.0,4.0), MCH=(30.0,1.5), MCHC=(33.0,0.8),
                      RDW=(14.5,1.5), PLT=(60,20), WBC=(2.5,0.8))
    else:  # Healthy
        params = dict(RBC=(5.0,0.5), HGB=(16.0,1.2), HCT=(42.0,2.0),
                      MCV=(90.0,5.0), MCH=(29.0,2.0), MCHC=(34.0,1.0),
                      RDW=(13.5,1.0), PLT=(280,40), WBC=(7.0,1.0))
    data = {k: np.random.normal(mu, sd, n_samples) for k,(mu,sd) in params.items()}
    data['Label'] = [label]*n_samples
    return pd.DataFrame(data)

# --- Generujemy dane i mieszamy ---
groups = ["Anemia Mikrocytarna","Anemia Makrocytarna","Anemia Normocytarna",
          "Anemia Hemolityczna","Anemia Aplastyczna","Healthy"]
df = pd.concat([generate_group_data(g) for g in groups],
               ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

# --- One-hot + normalizacja ---
df_encoded = pd.get_dummies(df, columns=['Label']).astype('float32')
label_cols = [c for c in df_encoded.columns if c.startswith('Label_')]
label_start_idx = df_encoded.columns.get_loc(label_cols[0])
numeric_cols = ['RBC','HGB','HCT','MCV','MCH','MCHC','RDW','PLT','WBC']
df_encoded[numeric_cols] = (df_encoded[numeric_cols] - df_encoded[numeric_cols].mean()) / df_encoded[numeric_cols].std()

# --- DataLoader dla VAE ---
data_tensor = torch.tensor(df_encoded.values, dtype=torch.float32)
dataset = TensorDataset(data_tensor)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
input_dim = df_encoded.shape[1]

# --- Definicja VAE ---
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim,64)
        self.fc2 = nn.Linear(64,32)
        self.fc31 = nn.Linear(32,latent_dim)
        self.fc32 = nn.Linear(32,latent_dim)
        self.fc4 = nn.Linear(latent_dim,32)
        self.fc5 = nn.Linear(32,64)
        self.fc6 = nn.Linear(64,input_dim)
        self.relu = nn.ReLU()
    def encode(self,x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)
    def reparameterize(self,mu,logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    def decode(self,z):
        h4 = self.relu(self.fc4(z))
        h5 = self.relu(self.fc5(h4))
        out = self.fc6(h5)
        out[:, label_start_idx:] = torch.sigmoid(out[:, label_start_idx:])
        return out
    def forward(self,x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

model = VAE(input_dim, latent_dim).to(device)
opt = optim.Adam(model.parameters(), lr=learning_rate)
mse_fn = nn.MSELoss(reduction='sum')

def loss_fn(recon, x, mu, logvar):
    mse = mse_fn(recon[:,:label_start_idx], x[:,:label_start_idx])
    bce = nn.BCEWithLogitsLoss()(recon[:,label_start_idx:], x[:,label_start_idx:])
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + bce + kld

# --- Trening ---
model.train()
for epoch in range(1, epochs+1):
    total = 0
    for batch in loader:
        x = batch[0].to(device)
        opt.zero_grad()
        recon, mu, logvar = model(x)
        loss = loss_fn(recon, x, mu, logvar)
        loss.backward()
        opt.step()
        total += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}  Avg Loss: {total/len(dataset):.4f}")

# --- Generacja (bez .numpy()) ---
model.eval()
with torch.no_grad():
    z = torch.randn(len(dataset), latent_dim).to(device)
    gen_tensor = model.decode(z).cpu().detach()       # tensor bez numpy()
    gen_list   = gen_tensor.tolist()                  # konwersja na listę Pythona

# Tworzymy DataFrame z listy
df_generated = pd.DataFrame(gen_list, columns=df_encoded.columns)
# Odwracanie normalizacji
for col in numeric_cols:
    m, s = df[col].mean(), df[col].std()
    df_generated[col] = df_generated[col]*s + m
df_generated.to_csv("dane-anemia.csv", index=False)
print("Zapisano syntetyczne dane w pliku dane-anemia.csv")
# --- Ocena jakości ---
X_real  = df[numeric_cols].values
X_synth = df_generated[numeric_cols].values
y_real  = np.zeros(len(X_real),  dtype=int)
y_synth = np.ones(len(X_synth), dtype=int)

X = np.vstack([X_real, X_synth])
y = np.concatenate([y_real, y_synth])

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_tr, y_tr)
proba = clf.predict_proba(X_te)[:,1]

# ROC + AUC
fpr, tpr, _ = roc_curve(y_te, proba)
roc_auc = auc(fpr, tpr)
plt.figure(); plt.plot(fpr,tpr,label=f'AUC={roc_auc:.2f}'); plt.plot([0,1],[0,1],'--')
plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC: Real vs Synthetic'); plt.legend(); plt.show()

# Confusion Matrix
y_pred = clf.predict(X_te)
cm = confusion_matrix(y_te, y_pred)
print("Confusion Matrix (rows=true, cols=pred):\n", cm)

# Korelacje cech
corr_real  = pd.DataFrame(X_real,  columns=numeric_cols).corr()
corr_synt  = pd.DataFrame(X_synth, columns=numeric_cols).corr()

plt.figure(); plt.imshow(corr_real, vmin=-1, vmax=1)
plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
plt.yticks(range(len(numeric_cols)), numeric_cols)
plt.title('Correlation Matrix – REAL'); plt.colorbar(); plt.show()

plt.figure(); plt.imshow(corr_synt, vmin=-1, vmax=1)
plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
plt.yticks(range(len(numeric_cols)), numeric_cols)
plt.title('Correlation Matrix – SYNTHETIC'); plt.colorbar(); plt.show()
