
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Ustawienia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
epochs = 100
learning_rate = 1e-3
latent_dim = 10  # wymiar przestrzeni latentnej


# Załóżmy, że masz DataFrame df z danymi, np. taki jak wcześniej:
# Kolumny: ['RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'RDW', 'PLT', 'WBC', 'Label']

# Przykładowa symulacja danych (analogiczna do poprzedniego przykładu)
def generate_group_data(label, n_samples=3000):
    if label == "Anemia Mikrocytarna":
        RBC = np.random.normal(4.0, 0.5, n_samples)
        HGB = np.random.normal(11.0, 1.0, n_samples)
        HCT = np.random.normal(33.0, 2.0, n_samples)
        MCV = np.random.normal(70.0, 5.0, n_samples)
        MCH = np.random.normal(23.0, 2.0, n_samples)
        MCHC = np.random.normal(32.0, 1.0, n_samples)
        RDW = np.random.normal(15.0, 1.5, n_samples)
        PLT = np.random.normal(300, 50, n_samples)
        WBC = np.random.normal(7.0, 1.0, n_samples)
    elif label == "Anemia Makrocytarna":
        RBC = np.random.normal(3.8, 0.5, n_samples)
        HGB = np.random.normal(10.5, 1.0, n_samples)
        HCT = np.random.normal(31.0, 2.0, n_samples)
        MCV = np.random.normal(105.0, 10.0, n_samples)
        MCH = np.random.normal(35.0, 3.0, n_samples)
        MCHC = np.random.normal(33.0, 1.0, n_samples)
        RDW = np.random.normal(16.0, 1.5, n_samples)
        PLT = np.random.normal(200, 30, n_samples)
        WBC = np.random.normal(7.0, 1.0, n_samples)
    elif label == "Anemia Mikrocytarna":
        RBC = np.random.normal(4.0, 0.5, n_samples)
        HGB = np.random.normal(11.0, 1.0, n_samples)
        HCT = np.random.normal(33.0, 2.0, n_samples)
        MCV = np.random.normal(90.0, 5.0, n_samples)
        MCH = np.random.normal(30.0, 2.0, n_samples)
        MCHC = np.random.normal(33.0, 1.0, n_samples)
        RDW = np.random.normal(15.0, 1.5, n_samples)
        PLT = np.random.normal(250, 40, n_samples)
        WBC = np.random.normal(8.0, 1.0, n_samples)
    else:
        RBC = np.random.normal(5.0, 0.5, n_samples)  # liczba erytrocytów (typowy zakres ~4.5-5.9 x10^6/µl)
        HGB = np.random.normal(16.0, 1.2,n_samples)  # hemoglobina (dla mężczyzn ~13.8-17.2 g/dl, dla kobiet ~12.1-15.1 g/dl)
        HCT = np.random.normal(42.0, 2.0, n_samples)  # hematokryt (mężczyźni ~40-50%, kobiety ~36-44%)
        MCV = np.random.normal(90.0, 5.0, n_samples)  # średnia objętość krwinki (80-100 fl)
        MCH = np.random.normal(29.0, 2.0, n_samples)  # średnia masa hemoglobiny w krwince (27-33 pg)
        MCHC = np.random.normal(34.0, 1.0, n_samples)  # średnie stężenie hemoglobiny (33-36 g/dl)
        RDW = np.random.normal(13.5, 1.0, n_samples)  # wskaźnik zróżnicowania wielkości krwinek (11.5-14.5%)
        PLT = np.random.normal(280, 40, n_samples)  # płytki krwi (150-450 x10^3/µl)
        WBC = np.random.normal(7.0, 1.0, n_samples)


    df = pd.DataFrame({
        'RBC': RBC,
        'HGB': HGB,
        'HCT': HCT,
        'MCV': MCV,
        'MCH': MCH,
        'MCHC': MCHC,
        'RDW': RDW,
        'PLT': PLT,
        'WBC': WBC,
        'Label': [label] * n_samples
    })
    return df


data_mikro = generate_group_data("Anemia Mikrocytarna")
data_makro = generate_group_data("Anemia Makrocytarna")
data_normo = generate_group_data("Anemia Normocytarna")
data_healthy = generate_group_data("Healthy")

df = pd.concat([data_mikro, data_makro, data_normo], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Przekształcamy kolumnę Label do postaci zmiennych zerojedynkowych (one-hot encoding)
df_encoded = pd.get_dummies(df, columns=['Label'])

# Normalizacja danych numerycznych (opcjonalnie, ale pomocna przy trenowaniu VAE)
numeric_cols = ['RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'RDW', 'PLT', 'WBC']
df_encoded[numeric_cols] = (df_encoded[numeric_cols] - df_encoded[numeric_cols].mean()) / df_encoded[numeric_cols].std()

# Konwersja wszystkich kolumn na typ float32
df_encoded = df_encoded.astype('float32')

# Przygotowujemy tensory
data_tensor = torch.tensor(df_encoded.values, dtype=torch.float32)

dataset = TensorDataset(data_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

input_dim = df_encoded.shape[1]


# Definicja modelu VAE
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc31 = nn.Linear(32, latent_dim)  # średnia
        self.fc32 = nn.Linear(32, latent_dim)  # logwariancja

        self.fc4 = nn.Linear(latent_dim, 32)
        self.fc5 = nn.Linear(32, 64)
        self.fc6 = nn.Linear(64, input_dim)

        self.relu = nn.ReLU()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h4 = self.relu(self.fc4(z))
        h5 = self.relu(self.fc5(h4))
        return self.fc6(h5)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


model = VAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
reconstruction_loss_fn = nn.MSELoss(reduction='sum')


# Funkcja utraty (loss) dla VAE
def loss_function(recon_x, x, mu, logvar):
    # Utrata rekonstrukcji
    recon_loss = reconstruction_loss_fn(recon_x, x)
    # KLD - odchylenie od rozkładu normalnego
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + KLD


# Trening modelu
model.train()
for epoch in range(1, epochs + 1):
    train_loss = 0
    for batch in dataloader:
        data = batch[0].to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {train_loss / len(dataset):.4f}")

# Generacja syntetycznych danych
model.eval()
with torch.no_grad():
    # Generujemy tyle próbek, ile mamy oryginalnych
    z = torch.randn(len(dataset), latent_dim).to(device)
    generated = model.decode(z).cpu().numpy()

# Odwracamy normalizację danych numerycznych
df_generated = pd.DataFrame(generated, columns=df_encoded.columns)
for col in numeric_cols:
    mean = df[col].mean()
    std = df[col].std()
    df_generated[col] = df_generated[col] * std + mean

print("Przykładowe wygenerowane dane:")
print(df_generated.head())

# Opcjonalnie: zapis do pliku CSV
df_generated.to_csv("synthetic_data_vae.csv", index=False)
