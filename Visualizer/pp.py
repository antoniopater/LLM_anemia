import os
import traceback
import joblib
import pandas as pd
import locale
import streamlit as st

locale.setlocale(locale.LC_NUMERIC, "pl_PL.UTF-8")

st.set_page_config(page_title="System predykcji chorób",
                   page_icon="❤️", layout="wide")

# Sidebar
st.sidebar.title("Wybór choroby")
selected = st.sidebar.radio("Wybierz chorobę:", ["Anemia", "Diabetes", "Infection"])

if selected == "Anemia":
    # mapowania
    label_mapping = {
        "Anemia Mikrocytarna": 0,
        "Anemia Makrocytarna": 1,
        "Anemia Normocytarna":2,
        "Anemia Hemolityczna":3,
        "Anemia Aplastyczna":4,
        "Healthy":5
    }
    typeMapping = {v:k for k,v in label_mapping.items()}

    features = ['RBC','HGB','HCT','MCV','MCH','MCHC','RDW','PLT','WBC']
    units = {
        'RBC':'×10⁶/µL','HGB':'g/dL','HCT':'%','MCV':'fL',
        'MCH':'pg','MCHC':'g/dL','RDW':'%','PLT':'×10³/µL',
        'WBC':'×10³/µL'
    }

    def to_float(x):
        try: return float(x.replace(",","."))
        except: return None

    def load_artifacts(model_fn="modelXGBoost.pkl",
                       scaler_fn="scaler.pkl"):
        base = os.path.dirname(os.path.abspath(__file__))
        m_p = os.path.join(base, model_fn)
        s_p = os.path.join(base, scaler_fn)
        for name,path in [("model",m_p),("scaler",s_p)]:
            if not os.path.exists(path):
                st.error(f"Nie znaleziono {name} pod {path}")
        model  = joblib.load(m_p)
        scaler = joblib.load(s_p)
        return model, scaler

    st.title("System predykcji anemii")
    model, scaler = load_artifacts()

    st.header("Dane pacjenta")
    cols = st.columns(3)
    inp  = {}
    for i,f in enumerate(features):
        col = cols[i%3]
        inp[f] = col.text_input(f"{f} ({units[f]})")

    if st.button("Predict"):
        data = {f: to_float(inp[f]) for f in features}
        df_i = pd.DataFrame([data])
        if df_i.isnull().any(axis=1).iloc[0]:
            st.warning("Uzupełnij wszystkie pola poprawnie")
        else:
            Xs = scaler.transform(df_i)
            pred     = model.predict(Xs)[0]
            prob_vec = model.predict_proba(Xs)[0]
            prob_pred= prob_vec[pred]

            res   = typeMapping[pred]
            ccol  = 'green' if res=='Healthy' else 'red'
            st.markdown(f"<h1 style='color:{ccol};'>{res}</h1>",
                        unsafe_allow_html=True)
            st.write(f"Prawdopodobieństwo ({res}): {prob_pred*100:.2f}%")

            # pełny rozkład
            pdf = pd.DataFrame({
                'Klasa':[typeMapping[i] for i in range(len(prob_vec))],
                'Prawd [%]':(prob_vec*100).round(2)
            })
            st.subheader("Pełny rozkład prawdopodobieństw")
            st.table(pdf)

elif selected in ["Diabetes","Infection"]:
    st.title(f"System predykcji: {selected}")
    st.info("Moduł w trakcie opracowywania.")
