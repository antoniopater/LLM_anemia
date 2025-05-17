import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# katalog, w którym leży ten plik
_tmp_dir = os.path.dirname(os.path.abspath(__file__))

# ścieżki do artefaktów
tmp_dir = _tmp_dir
SCALER_PATH = os.path.join(tmp_dir, 'scaler.pkl')
LE_PATH     = os.path.join(tmp_dir, 'label_encoder.pkl')


def changeLabels(csv_path=None, output_csv=None):
    """
    Wczytuje CSV z danymi syntetycznymi lub oryginalnymi zawierającymi kolumny one-hot:
    ['Label_Anemia Aplastyczna', 'Label_Anemia Hemolityczna',
     'Label_Anemia Makrocytarna', 'Label_Anemia Mikrocytarna',
     'Label_Anemia Normocytarna', 'Label_Healthy']
    Tworzy kolumnę 'Label_argmax' z maksymalnej kolumny one-hot,
    tworzy 'Label' (tekstową) i 'Label_num' (liczbową), zapisuje LabelEncoder,
    oraz opcjonalnie zapisuje zmodyfikowany DataFrame do output_csv.
    Zwraca DataFrame z kolumnami cech + ['Label','Label_num'].
    """
    if csv_path is None:
        csv_path = os.path.join(tmp_dir, 'dane-anemia.csv')
    df = pd.read_csv(csv_path)

    # wykryj wszystkie kolumny one-hot label
    label_cols = [c for c in df.columns if c.startswith('Label_')]
    if not label_cols:
        raise ValueError("Brak kolumn one-hot zaczynających się od 'Label_'")

    # znajdź najbardziej prawdopodobną etykietę
    df['Label_argmax'] = df[label_cols].idxmax(axis=1)
    # usuń prefiks
    df['Label'] = df['Label_argmax'].str.replace('Label_', '')

    # enkodowanie etykiet
    le = LabelEncoder()
    df['Label_num'] = le.fit_transform(df['Label'])
    # zapisz encoder
    joblib.dump(le, LE_PATH)

    # opcjonalne zapisanie rozszerzonego pliku
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Zapisano zmodyfikowany DataFrame do {output_csv}")

    # zwróć tylko cechy + Label + Label_num
    features = ['RBC','HGB','HCT','MCV','MCH','MCHC','RDW','PLT','WBC', 'Label', 'Label_num']
    return df[features]


def scaleData(X, output=True):
    """
    Fit-transform StandardScaler na X (array lub DataFrame.values),
    zapisuje scaler.pkl i zwraca X_scaled.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_PATH)
    if output:
        print(f"Zapisano scaler do {SCALER_PATH}")
    return X_scaled


def transformData(X):
    """
    Transformuje X na podstawie zapisanego scaler.pkl (tylko transform).
    """
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Nie znaleziono pliku scaler.pkl w {tmp_dir}")
    scaler = joblib.load(SCALER_PATH)
    return scaler.transform(X)


if __name__ == '__main__':
    # przykład użycia
    df_pre = changeLabels(csv_path=os.path.join(tmp_dir, 'dane-anemia.csv'),
                          output_csv=os.path.join(tmp_dir, 'dane-anemia-labeled.csv'))
    X = df_pre[['RBC','HGB','HCT','MCV','MCH','MCHC','RDW','PLT','WBC']].values
    Xs = scaleData(X)
    print(f"Przetworzono {len(df_pre)} rekordów, X_scaled.shape = {Xs.shape}")
