import numpy as np
import pandas as pd

# Liczba próbek na każdą kategorię
n_per_category = 3333
total_samples = n_per_category * 4

def generate_normal_values(n):
    # Generowanie wartości dla parametrów krwi z użyciem rozkładów normalnych i ograniczeń
    WBC = np.clip(np.random.normal(loc=7.5, scale=1.5, size=n), 4, 11)

    NEUT = np.clip(np.random.normal(loc=4, scale=1, size=n), 2, 7)
    LYMPH = np.clip(np.random.normal(loc=2.5, scale=0.5, size=n), 1, 4)
    ALY = np.clip(np.random.normal(loc=0.5, scale=0.2, size=n), 0.2, 1)
    ASLY = np.clip(np.random.normal(loc=0.5, scale=0.2, size=n), 0.2, 1)
    MONO = np.clip(np.random.normal(loc=0.5, scale=0.1, size=n), 0.3, 1)
    EOS = np.clip(np.random.normal(loc=0.1, scale=0.05, size=n), 0.05, 0.3)
    BASO = np.clip(np.random.normal(loc=0.1, scale=0.05, size=n), 0.05, 0.3)

    # Procentowe udziały
    NEUT_perc = np.clip((NEUT / WBC) * 100 + np.random.normal(0, 2, n), 40, 80)
    LYMPH_perc = np.clip((LYMPH / WBC) * 100 + np.random.normal(0, 2, n), 15, 40)
    ALY_perc = np.clip((ALY / WBC) * 100 + np.random.normal(0, 1, n), 5, 15)
    ASLY_perc = np.clip((ASLY / WBC) * 100 + np.random.normal(0, 1, n), 5, 15)
    MONO_perc = np.clip((MONO / WBC) * 100 + np.random.normal(0, 1, n), 2, 10)
    EOS_perc = np.clip((EOS / WBC) * 100 + np.random.normal(0, 1, n), 0.5, 5)
    BASO_perc = np.clip((BASO / WBC) * 100 + np.random.normal(0, 1, n), 0.5, 5)

    # NEUT-GI i NEUT-RI – parametry kategoryczne
    NEUT_GI = np.random.choice([0, 1], size=n)
    NEUT_RI = np.random.choice([0, 1], size=n)

    # Parametry erytrocytów
    RBC = np.clip(np.random.normal(loc=5, scale=0.5, size=n), 4, 6)
    HGB = np.clip(np.random.normal(loc=15, scale=1.5, size=n), 12, 17)
    HCT = HGB * 3  # uproszczone przeliczenie hematokrytu
    MCV = np.clip(np.random.normal(loc=90, scale=5, size=n), 80, 100)
    MCH = np.clip(np.random.normal(loc=30, scale=2, size=n), 25, 35)
    MCHC = np.clip(np.random.normal(loc=34, scale=1, size=n), 32, 36)
    RDW_SD = np.clip(np.random.normal(loc=42, scale=2, size=n), 39, 46)
    RDW_CV = np.clip(np.random.normal(loc=13, scale=1, size=n), 11.5, 14.5)

    # Parametry wielkości erytrocytów (procentowo)
    Mikrocyty = np.random.uniform(0, 2, n)
    Makrocyty = np.random.uniform(0, 2, n)

    # Erytroblasty
    NRBC = np.clip(np.random.normal(loc=0.05, scale=0.03, size=n), 0, 0.1)
    NRBC_perc = np.clip(np.random.normal(loc=0.05, scale=0.03, size=n), 0, 0.1)

    # Parametry płytek krwi
    PLT = np.clip(np.random.normal(loc=250, scale=50, size=n), 150, 450)
    PDW = np.clip(np.random.normal(loc=13, scale=2, size=n), 10, 17)
    MPV = np.clip(np.random.normal(loc=9, scale=1, size=n), 7.5, 11)
    PLCR = np.clip(np.random.normal(loc=25, scale=5, size=n), 15, 35)
    PCT = np.clip(np.random.normal(loc=0.3, scale=0.05, size=n), 0.19, 0.39)

    return {
        "WBC": WBC,
        "NEUT#": NEUT,
        "LYMPH#": LYMPH,
        "ALY#": ALY,
        "ASLY#": ASLY,
        "MONO#": MONO,
        "EOS#": EOS,
        "BASO#": BASO,
        "NEUT%": NEUT_perc,
        "LYMPH%": LYMPH_perc,
        "ALY%": ALY_perc,
        "ASLY%": ASLY_perc,
        "MONO%": MONO_perc,
        "EOS%": EOS_perc,
        "BASO%": BASO_perc,
        "NEUT-GI": NEUT_GI,
        "NEUT-RI": NEUT_RI,
        "RBC": RBC,
        "HGB": HGB,
        "HCT": HCT,
        "MCV": MCV,
        "MCH": MCH,
        "MCHC": MCHC,
        "RDW-SD": RDW_SD,
        "RDW-CV": RDW_CV,
        "Mikrocyty": Mikrocyty,
        "Makrocyty": Makrocyty,
        "NRBC#": NRBC,
        "NRBC%": NRBC_perc,
        "PLT": PLT,
        "PDW": PDW,
        "MPV": MPV,
        "PLCR": PLCR,
        "PCT": PCT,
    }

# Generowanie danych bazowych
data = generate_normal_values(total_samples)
df = pd.DataFrame(data)

# Przydzielenie etykiety typu anemii w równych ilościach
anemia_types = (
    ["normal"] * n_per_category +
    ["anemia_microcytic"] * n_per_category +
    ["anemia_macrocytic"] * n_per_category +
    ["anemia_normocytic"] * n_per_category
)
anemia_types = np.array(anemia_types)
np.random.shuffle(anemia_types)
df["anemia_type"] = anemia_types

# Modyfikacja parametrów wg kluczowych wzorców

# A) Anemia mikrocytarna: ↓ MCV, ↓ MCH, ↑ RDW
micro_idx = df["anemia_type"] == "anemia_microcytic"
df.loc[micro_idx, "MCV"] *= np.random.uniform(0.8, 0.9, micro_idx.sum())  # zmniejszenie MCV
df.loc[micro_idx, "MCH"] *= np.random.uniform(0.8, 0.9, micro_idx.sum())  # zmniejszenie MCH
df.loc[micro_idx, "RDW-CV"] *= np.random.uniform(1.2, 1.3, micro_idx.sum())  # zwiększenie RDW

# B) Anemia makrocytarna: ↑ MCV, ↑ MCH, ↓ PLT
macro_idx = df["anemia_type"] == "anemia_macrocytic"
df.loc[macro_idx, "MCV"] *= np.random.uniform(1.1, 1.3, macro_idx.sum())  # zwiększenie MCV
df.loc[macro_idx, "MCH"] *= np.random.uniform(1.1, 1.3, macro_idx.sum())  # zwiększenie MCH
df.loc[macro_idx, "PLT"] *= np.random.uniform(0.8, 0.9, macro_idx.sum())  # zmniejszenie liczby płytek

# C) Anemia normocytarna: ↓ RBC, MCV bez zmian, ↑ RDW
normo_idx = df["anemia_type"] == "anemia_normocytic"
df.loc[normo_idx, "RBC"] *= np.random.uniform(0.8, 0.9, normo_idx.sum())  # zmniejszenie RBC
df.loc[normo_idx, "RDW-CV"] *= np.random.uniform(1.2, 1.3, normo_idx.sum())  # zwiększenie RDW

# Zapis danych do pliku CSV
df.to_csv("medical_data_anemia_patterns.csv", index=False)
print("Plik CSV 'medical_data_anemia_patterns.csv' został wygenerowany z łączną liczbą rekordów:", total_samples)
