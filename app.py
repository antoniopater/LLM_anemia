import numpy as np
import streamlit as st
import pandas as pd
import joblib
import locale

locale.setlocale(locale.LC_NUMERIC, "pl_PL.UTF-8")

label_mapping = {
    "Anemia Mikrocytarna": 0,
    "Anemia Makrocytarna": 1,
    "Anemia Normocytarna": 2,
    "Healthy": 3
}
features = ['RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'RDW', 'PLT', 'WBC']
typeMapping = {v: k for k, v in label_mapping.items()}

st.set_page_config(page_title="Predykcja rodzaju anemii", page_icon="❤️")


def convert_to_float(value):
    try:
        return float(value.replace(",", "."))
    except ValueError:
        return None


def load_model(modelPath="./XGBoost_models/xgb_model.pkl", scalerPath="./KNN_with_better_preprocess/scaler.pkl",
               pcaPath="./KNN_with_better_preprocess/pca.pkl"):
    try:
        model = joblib.load(modelPath)
        scaler = joblib.load(scalerPath)
        pca = joblib.load(pcaPath)
        return model, scaler, pca
    except Exception as e:
        print("Blad ladowania", e)
    return None, None, None


try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

st.title("System predykcji anemii")

api_key = None
gemini_model = None

if GEMINI_AVAILABLE:
    with st.sidebar:
        st.header("Ustawienia LLM")
        api_key = st.text_input("Klucz API Gemini", type="password")
        if api_key:
            try:
                genai.configure(api_key=api_key)
                gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                st.success("Połączono z Gemini API")
            except Exception as e:
                st.error(f"Błąd Gemini: {e}")
                gemini_model = None

model, scaler, pca = load_model()

st.header("Dane pacjenta")
col1, col2, col3 = st.columns(3)

with col1:
    RBC = st.text_input("RBC")
    HGB = st.text_input("HGB")
    HCT = st.text_input("HCT")

with col2:
    MCV = st.text_input("MCV")
    MCH = st.text_input("MCH")
    MCHC = st.text_input("MCHC")

with col3:
    PLT = st.text_input("PLT")
    WBC = st.text_input("WBC")
    RDW = st.text_input("RDW")

if st.button("Predict", type="primary"):
    input_data = {
        'RBC': convert_to_float(RBC),
        'HGB': convert_to_float(HGB),
        'HCT': convert_to_float(HCT),
        'MCV': convert_to_float(MCV),
        'MCH': convert_to_float(MCH),
        'MCHC': convert_to_float(MCHC),
        'RDW': convert_to_float(RDW),
        'PLT': convert_to_float(PLT),
        'WBC': convert_to_float(WBC)
    }
    exceptionOcurred = False

    try:
        input_df = pd.DataFrame([input_data])
        input_df = input_df[features]
        input_scaled = scaler.transform(input_df)
        input_pca = pca.transform(input_scaled)
        prediction = model.predict(input_pca)
        probability = model.predict_proba(input_pca)[0][int(prediction)]
    except ValueError:
        st.info("Proszę o poprawne uzupełniene formularza")
        exceptionOcurred = True
    # Predykcja

    if not exceptionOcurred:

        st.markdown("---")
        st.header("Wyniki predykcji")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Predykcja")
            if isinstance(prediction, np.ndarray):
                prediction = prediction.item()  # Extract scalar from the array
            result = typeMapping[int(prediction)]

            try:

                formatted_prediction = f"<h1 style='text-align: center; color: {'red' if result != 'Healthy' else 'green'};'>{result}</h1>"

            except (ValueError, TypeError) as e:

                formatted_prediction = "<h1>Error: Invalid result format</h1>"
                formatted_probability = "<h1>Error: Invalid result format</h1>"
                print(e)

            try:
                formatted_probability = f"<h1 style='text-align: center; color: {'red' if probability < 0.5 else 'green'};'>Prawdopodobieństwo: {probability * 100:.2f}%</h1>"
            except (ValueError, TypeError) as e:
                # Catch cases where result cannot be converted to float or has other issues
                formatted_probability = "<h1>Error: Invalid result format</h1>"

            st.markdown(
                formatted_prediction,
                unsafe_allow_html=True)
            st.markdown(
                formatted_probability,
                unsafe_allow_html=True)

        with col2:
            st.subheader("Interpretacja")
            prawdSlownie = 'wysokie' if probability > 0.5 else 'niskie'
            print(prediction, prawdSlownie)

            if prediction != 'Healthy':
                st.error(
                    f"Model przewiduje {prawdSlownie} ryzyko ze u pacjenta występuje choroba {typeMapping[prediction]}.")
            else:
                st.error(
                    f"Model przewiduje {prawdSlownie} prawdopodobienstwo ze pacjent jest zdrowy.")

            st.info("Pamiętaj, że jest to tylko wskazówka od modelu AI i powinna być zawsze weryfikowana przez lekarza.")

        if GEMINI_AVAILABLE and gemini_model:
            st.markdown('---')
            st.header("Epikryza medyczna")


            with st.spinner("Generowanie epikryzy..."):
                prompt = f"""
                Jako hematolog, na podstawie danych pacjenta (RBC: {RBC}, HGB: {HGB}, HCT: {HCT}, MCV: {MCV}, MCH: {MCH},
                MCHC: {MCHC}, RDW: {RDW}, PLT: {PLT}, WBC: {WBC})
                oraz przewidywania modelu AI: {'pacjent zdrowy' if typeMapping[prediction] == 'Healthy' else 'pacjent choruje na '+ typeMapping[prediction]} 
                prawdopodobieństwo: {probability * 100:.2f}% napisz profesjonalną epikryzę medyczną w języku polskim, 
                w następującym schemacie:
                PACJENT ...
                PARAMETRY ŻYCIOWE ...
                WYKONANE BADANIA ...
                ROZPOZNANIE ...
                PRZEWIDYWANY SKUTEK CHOROBY ...
                ZALECENIA ...
                
                Uwaga - nie pisz na końcu uwagi, że jest to przykładowa epikryza, ja o tym wiem. 
                Robię teraz po prostu doświadczenie. Mam świadomość, że to nie są prawdziwe dane, ani zalecenia, nie ostrzegaj mnie o tym.
                """

                try:
                    response = gemini_model.generate_content(prompt)
                    st.markdown(response.text)

                    from datetime import datetime

                    current_date = datetime.now().strftime("%Y-%m-%d")
                    epicrisis_filename = f"epikryza_{current_date}.txt"
                    st.download_button(
                        label = "Pobierz epikryzę",
                        data = response.text,
                        file_name=epicrisis_filename,
                        mime = 'text/plain',
                    )
                except Exception as e:
                    st.error("Błąd generowania epikryzy")
        elif GEMINI_AVAILABLE and api_key is None:
            st.markdown('---')
            st.header("Epikryza medyczna")
            st.info("Wprowadz klucz API Gemini w panelu bocznym, aby wygenerować epikryzę.")