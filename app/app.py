import os
import traceback

import joblib
import pandas as pd
import locale
import streamlit as st
from dotenv import load_dotenv
from fpdf import FPDF
import re

locale.setlocale(locale.LC_NUMERIC, "pl_PL.UTF-8")

st.set_page_config(page_title="System predykcji chorób", page_icon="❤️", layout="wide")

# Sidebar - Wybór choroby
st.sidebar.title("Wybór choroby")
selected_disease = st.sidebar.radio("Wybierz chorobę:", ["Anemia", "Diabetes", "Infection"])

if selected_disease == "Anemia":
    label_mapping = {
        "Anemia Mikrocytarna": 0,
        "Anemia Makrocytarna": 1,
        "Anemia Normocytarna": 2,
        "Anemia Hemolityczna":3,
        "Anemia Aplastyczna":4,
        "Healthy": 5
    }
    features = ['RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'RDW', 'PLT', 'WBC']
    typeMapping = {v: k for k, v in label_mapping.items()}

    units = {
        'RBC': '×10⁶/µL',
        'HGB': 'g/dL',
        'HCT': '%',
        'MCV': 'fL',
        'MCH': 'pg',
        'MCHC': 'g/dL',
        'RDW': '%',
        'PLT': '×10³/µL',
        'WBC': '×10³/µL'
    }

    def convert_to_float(value):
        try:
            return float(value.replace(",", "."))
        except ValueError:
            return None


    def load_model(
            modelPath="models/anemia/modelXGBoost.pkl",
            scalerPath="preprocess/anemia/scaler.pkl",
            pcaPath="preprocess/anemia/pca.pkl"
    ):
        try:
            model_full_path = os.path.abspath(modelPath)
            scaler_full_path = os.path.abspath(scalerPath)
            pca_full_path = os.path.abspath(pcaPath)

            # Weryfikacja, czy pliki istnieją
            if not os.path.exists(model_full_path):
                print(f"❌ Nie znaleziono modelu pod ścieżką: {model_full_path}")
            if not os.path.exists(scaler_full_path):
                print(f"❌ Nie znaleziono scalera pod ścieżką: {scaler_full_path}")
            if not os.path.exists(pca_full_path):
                print(f"❌ Nie znaleziono PCA pod ścieżką: {pca_full_path}")

            model = joblib.load(model_full_path)
            scaler = joblib.load(scaler_full_path)
            pca = joblib.load(pca_full_path)

            print("✅ Model, scaler i PCA zostały poprawnie załadowane.")
            return model, scaler, pca

        except Exception as e:
            print("❌ Błąd ładowania modelu lub preprocesorów:")
            traceback.print_exc()
            return None, None, None

    st.title("System predykcji anemii")

    model, scaler, pca = load_model()

    # Interfejs użytkownika
    st.header("Dane pacjenta")
    col1, col2, col3 = st.columns(3)

    with col1:
        RBC = st.text_input(f"RBC ({units['RBC']})")
        HGB = st.text_input(f"HGB ({units['HGB']})")
        HCT = st.text_input(f"HCT ({units['HCT']})")

    with col2:
        MCV = st.text_input(f"MCV ({units['MCV']})")
        MCH = st.text_input(f"MCH ({units['MCH']})")
        MCHC = st.text_input(f"MCHC ({units['MCHC']})")

    with col3:
        PLT = st.text_input(f"PLT ({units['PLT']})")
        WBC = st.text_input(f"WBC ({units['WBC']})")
        RDW = st.text_input(f"RDW ({units['RDW']})")

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

        try:
            input_df = pd.DataFrame([input_data])
            input_df = input_df[features]
            input_scaled = scaler.transform(input_df)
            input_pca = pca.transform(input_scaled)
            prediction = model.predict(input_pca)
            probability = model.predict_proba(input_pca)[0][int(prediction)]
        except ValueError:
            st.info("Proszę o poprawne uzupełnienie formularza")
        else:
            st.markdown("---")
            st.header("Wyniki predykcji")
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Predykcja")
                result = typeMapping[int(prediction)]
                formatted_prediction = f"<h1 style='text-align: center; color: {'red' if result != 'Healthy' else 'green'};'>{result}</h1>"
                formatted_probability = f"<h1 style='text-align: center; color: {'red' if probability < 0.5 else 'green'};'>Prawdopodobieństwo: {probability * 100:.2f}%</h1>"
                st.markdown(formatted_prediction, unsafe_allow_html=True)
                st.markdown(formatted_probability, unsafe_allow_html=True)

            with col2:
                st.subheader("Interpretacja")
                prawdSlownie = 'wysokie' if probability > 0.5 else 'niskie'

                if result != 'Healthy':
                    st.error(f"Model przewiduje {prawdSlownie} ryzyko, że pacjent choruje na {result}.")
                else:
                    st.success(f"Model przewiduje {prawdSlownie} prawdopodobieństwo, że pacjent jest zdrowy.")

                st.info("Pamiętaj, że jest to tylko wskazówka od modelu AI i powinna być zawsze weryfikowana przez lekarza.")

            # Generowanie epikryzy z Gemini
            try:
                import google.generativeai as genai

                load_dotenv()
                api_key = os.getenv("GEMINI_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                    st.markdown('---')
                    st.header("Epikryza medyczna")

                    with st.spinner("Generowanie epikryzy..."):
                        prompt = f"""
                        Jako hematolog, na podstawie danych pacjenta (RBC: {RBC} {units['RBC']}, HGB: {HGB} {units['HGB']}, 
                        HCT: {HCT} {units['HCT']}, MCV: {MCV} {units['MCV']}, MCH: {MCH} {units['MCH']}, 
                        MCHC: {MCHC} {units['MCHC']}, RDW: {RDW} {units['RDW']}, PLT: {PLT} {units['PLT']}, 
                        WBC: {WBC} {units['WBC']})
                        oraz przewidywania modelu AI: {'pacjent zdrowy' if result == 'Healthy' else 'pacjent choruje na ' + result} 
                        prawdopodobieństwo: {probability * 100:.2f}% napisz profesjonalną epikryzę medyczną w języku polskim.
                        """
                        response = gemini_model.generate_content(prompt)

                        # Usuwamy gwiazdki do formatowania w markdown
                        clean_response = response.text.replace("**", "")

                        # Usuwamy placeholdery w nawiasach kwadratowych
                        clean_response = re.sub(r'\[.*?]', '', clean_response)

                        st.markdown(clean_response)

                        # Generowanie PDF
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_font("Arial", size=12)

                        # Ładowanie czcionki DejaVuSans
                        pdf.add_font('DejaVu', '', 'fonts/DejaVuSans.ttf', uni=True)
                        pdf.set_font("DejaVu", size=12)

                        # Dodanie tekstu do PDF
                        pdf.multi_cell(0, 10, clean_response)

                        # Udostępnienie do pobrania (przycisk nie zniknie)
                        st.download_button(
                            label="Pobierz epikryzę",
                            data=pdf.output(dest="S").encode("latin-1"),
                            file_name="epikryza.pdf",
                            mime="application/pdf"
                        )

            except Exception as e:
                st.error("Błąd generowania epikryzy")
                st.error(e)
elif selected_disease in ["Diabetes", "Infection"]:
    st.title(f"System predykcji: {selected_disease}")
    st.info("Moduł dla tej choroby jest w trakcie opracowywania.")