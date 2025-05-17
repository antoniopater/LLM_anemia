# app.py
import os
import joblib
import pandas as pd
import locale
import streamlit as st

# ustaw locale dla liczb
try:
    locale.setlocale(locale.LC_NUMERIC, 'pl_PL.UTF-8')
except locale.Error:
    pass  # jeżeli locale nie jest dostępne

st.set_page_config(
    page_title='System predykcji chorób',
    page_icon='❤️',
    layout='wide'
)

# Sidebar
st.sidebar.title('Wybór choroby')
selected = st.sidebar.radio('Wybierz chorobę:', ['Anemia', 'Diabetes', 'Infection'])

if selected == 'Anemia':
    st.title('System predykcji anemii')

    # ścieżki do artefaktów
    base_dir = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(base_dir, 'modelXGBoost.pkl')
    SCALER_PATH = os.path.join(base_dir, 'scaler.pkl')
    LE_PATH     = os.path.join(base_dir, 'label_encoder.pkl')

    # wczytanie modelu, scaler i label encoder
    @st.cache_resource
    def load_artifacts():
        if not os.path.exists(MODEL_PATH):
            st.error(f"Nie znaleziono modelu pod {MODEL_PATH}")
        if not os.path.exists(SCALER_PATH):
            st.error(f"Nie znaleziono scaler.pkl pod {SCALER_PATH}")
        if not os.path.exists(LE_PATH):
            st.error(f"Nie znaleziono label_encoder.pkl pod {LE_PATH}")
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        le = joblib.load(LE_PATH)
        return model, scaler, le

    model, scaler, le = load_artifacts()

    # definicja cech i jednostek
    features = ['RBC','HGB','HCT','MCV','MCH','MCHC','RDW','PLT','WBC']
    units = {
        'RBC': '×10⁶/µL','HGB': 'g/dL','HCT': '%','MCV': 'fL',
        'MCH': 'pg','MCHC': 'g/dL','RDW': '%','PLT': '×10³/µL',
        'WBC': '×10³/µL'
    }

    st.header('Dane pacjenta')
    cols = st.columns(3)
    inputs = {}
    for i, f in enumerate(features):
        col = cols[i % 3]
        inputs[f] = col.text_input(f + ' (' + units[f] + ')')

    def to_float(x):
        try:
            return float(x.replace(',', '.'))
        except:
            return None

    if st.button('Predict'):
        # przygotuj DataFrame
        data = {f: to_float(inputs[f]) for f in features}
        df_i = pd.DataFrame([data])
        if df_i.isnull().any(axis=1).iloc[0]:
            st.warning('Uzupełnij wszystkie pola poprawnie')
        else:
            Xs = scaler.transform(df_i.values)
            pred_num = model.predict(Xs)[0]
            prob_vec = model.predict_proba(Xs)[0]
            prob = prob_vec[pred_num]

            # odwróć etykietę
            res = le.inverse_transform([pred_num])[0]
            color = 'green' if res == 'Healthy' else 'red'
            st.markdown(f"<h1 style='color:{color};'>{res}</h1>", unsafe_allow_html=True)
            st.write(f"Prawdopodobieństwo ({res}): {prob*100:.2f}%")

            # pełny rozkład
            dist_df = pd.DataFrame({
                'Klasa': le.inverse_transform(list(range(len(prob_vec)))),  # kolejno klasy
                'Prawdopodobieństwo [%]': (prob_vec*100).round(2)
            })
            st.subheader('Pełny rozkład prawdopodobieństw')
            st.table(dist_df)

elif selected in ['Diabetes', 'Infection']:
    st.title(f'System predykcji: {selected}')
    st.info('Moduł w trakcie opracowywania.')