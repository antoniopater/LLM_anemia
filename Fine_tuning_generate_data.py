import json
import random

def generate_age():
    return random.randint(20, 80)

def generate_gender():
    return random.choice(["Pacjent", "Pacjentka"])

def generate_lab_value(param, trend):
    if trend == "↓":
        return f"{param}: {round(random.uniform(0.5, 0.9), 2)}"  # Wartości poniżej normy
    elif trend == "↑":
        return f"{param}: {round(random.uniform(1.1, 1.5), 2)}"  # Wartości powyżej normy
    else:
        return f"{param}: {round(random.uniform(0.9, 1.1), 2)}"  # Wartości w normie

def generate_text_style(text):
    styles = [
        lambda t: t,  # Standardowy styl
        lambda t: t.replace(".", ";"),  # Użycie średników zamiast kropek
        lambda t: t.upper(),  # Wszystkie litery wielkie
        lambda t: t.replace(",", " ,")  # Dodanie spacji przed przecinkami
    ]
    style = random.choice(styles)
    return style(text)

def generate_microcytic_example(idx):
    age = generate_age()
    gender = generate_gender()
    lab_results = [
        generate_lab_value("RBC", "↓"),
        generate_lab_value("HGB", "↓"),
        generate_lab_value("HCT", "↓"),
        generate_lab_value("MCV", "↓"),
        generate_lab_value("MCH", "↓"),
        generate_lab_value("MCHC", "↓"),
        generate_lab_value("RDW", "↑"),
        "Mikrocyty: ↑",
        generate_lab_value("PLT", "↑")
    ]
    base_text = (f"{gender}, {age} lat, z objawami zmęczenia i osłabienia. "
                 f"Badania laboratoryjne (przykład {idx}): {', '.join(lab_results)}.")
    styled_text = generate_text_style(base_text)
    return {"input_text": styled_text, "label": "microcytic"}

def generate_macrocytic_example(idx):
    age = generate_age()
    gender = generate_gender()
    lab_results = [
        generate_lab_value("RBC", "↓"),
        generate_lab_value("HGB", "↓"),
        generate_lab_value("HCT", "↓"),
        generate_lab_value("MCV", "↑"),
        generate_lab_value("MCH", "↑"),
        generate_lab_value("MCHC", "N"),
        generate_lab_value("RDW", "↑"),
        "Makrocyty: ↑",
        generate_lab_value("PLT", "↓"),
        generate_lab_value("NRBC", "↑"),
        generate_lab_value("NEUT#", "↓")
    ]
    base_text = (f"{gender}, {age} lat, z objawami neuropatii i zaburzeniami pamięci. "
                 f"Wyniki badań (przykład {idx}): {', '.join(lab_results)}.")
    styled_text = generate_text_style(base_text)
    return {"input_text": styled_text, "label": "macrocytic"}

def generate_normocytic_example(idx):
    age = generate_age()
    gender = generate_gender()
    lab_results = [
        generate_lab_value("RBC", "↓"),
        generate_lab_value("HGB", "↓"),
        generate_lab_value("HCT", "↓"),
        generate_lab_value("MCV", "N"),
        generate_lab_value("MCH", "N"),
        generate_lab_value("MCHC", "N"),
        generate_lab_value("RDW", "↑"),
        generate_lab_value("NRBC", "↑"),
        generate_lab_value("Retikulocyty", "↑"),
        generate_lab_value("WBC", "↑"),
        generate_lab_value("NEUT#", "↑"),
        generate_lab_value("PLT", "↑")
    ]
    base_text = (f"{gender}, {age} lat, z przewlekłymi stanami zapalnymi i ogólnym osłabieniem. "
                 f"Badania laboratoryjne (przykład {idx}): {', '.join(lab_results)}.")
    styled_text = generate_text_style(base_text)
    return {"input_text": styled_text, "label": "normocytic"}

sample_examples = []

# Generujemy 500 przykładów dla każdej kategorii
for i in range(1, 501):
    sample_examples.append(generate_microcytic_example(i))
for i in range(1, 501):
    sample_examples.append(generate_macrocytic_example(i))
for i in range(1, 501):
    sample_examples.append(generate_normocytic_example(i))

# Tworzymy finalną strukturę JSON
data = {
    "fine_tuning_data": {
        "parameters": [
            {
                "name": "WBC",
                "description": "Liczba białych krwinek w krwi, odpowiadają za odporność organizmu.",
                "unit": "x10^3/μL"
            },
            {
                "name": "NEUT#",
                "description": "Liczba neutrofili, czyli granulocytów odpowiedzialnych za walkę z infekcjami bakteryjnymi.",
                "unit": "x10^3/μL"
            },
            # ... (pozostałe definicje parametrów)
        ],
        "advanced_medical_knowledge": {
            "anemia": {
                "microcytic": {
                    "description": "Anemia mikrocytarna, często związana z niedoborem żelaza lub talasemią.",
                    "key_features": {
                        "RBC": "↓ lub N (w talasemii może być N lub nawet ↑)",
                        "HGB": "↓",
                        "HCT": "↓",
                        "MCV": "↓",
                        "MCH": "↓",
                        "MCHC": "↓",
                        "RDW-SD_RDW-CV": "↑ (w niedoborze żelaza), N (w talasemii)",
                        "Mikrocyty": "↑",
                        "PLT": "często ↑",
                        "WBC": "zazwyczaj N",
                        "NEUT-GI_NEUT-RI": "zazwyczaj N"
                    }
                },
                "macrocytic": {
                    "description": "Anemia makrocytarna, najczęściej związana z niedoborem witaminy B12, kwasu foliowego lub chorobami wątroby.",
                    "key_features": {
                        "RBC": "↓",
                        "HGB": "↓",
                        "HCT": "↓",
                        "MCV": "↑",
                        "MCH": "↑",
                        "MCHC": "N",
                        "RDW-SD_RDW-CV": "↑",
                        "Makrocyty": "↑",
                        "PLT": "↓ (zwłaszcza w niedoborze B12)",
                        "NRBC#_NRBC%": "może być ↑ (w zaawansowanych przypadkach)",
                        "NEUT#": "może być ↓",
                        "NEUT-GI_NEUT-RI": "często ↓"
                    }
                },
                "normocytic": {
                    "description": "Anemia normocytarna, typowa dla anemii chorób przewlekłych lub hemolitycznej.",
                    "key_features": {
                        "RBC": "↓",
                        "HGB": "↓",
                        "HCT": "↓",
                        "MCV": "N",
                        "MCH": "N",
                        "MCHC": "N",
                        "RDW-SD_RDW-CV": "↑ (w anemii hemolitycznej)",
                        "NRBC#_NRBC%": "↑ (w nasilonej hemolizie)",
                        "Retikulocyty": "↑ w hemolizie, ↓ w anemii przewlekłej",
                        "WBC": "może być ↑ w stanach zapalnych",
                        "NEUT#": "może być ↑ w anemii przewlekłej",
                        "PLT": "może być ↑ w przewlekłych stanach zapalnych"
                    }
                }
            }
        },
        "sample_recommendation": {
            "general_guideline": "W przypadku fine-tuningu modelu medycznego kluczowa jest jakość oraz różnorodność danych. Zaleca się zebranie minimum 500 przykładów na każdą kategorię anemii, co daje łącznie około 1500 przykładów.",
            "suggested_sample_numbers": {
                "per_category": "Minimum 500 przykładów dla mikrocytarnej, makrocytarnej i normocytarnej anemii.",
                "total": "Około 1500 przykładów – im więcej, tym lepiej."
            },
            "notes": "W przypadku ograniczonej liczby danych, warto zastosować techniki augmentacji lub pozyskać dodatkowe źródła danych."
        },
        "training_config": {
            "epochs": 10,
            "learning_rate": 0.0001,
            "batch_size": 16,
            "optimizer": "adam",
            "scheduler": "None"
        },
        "data_split": {
            "train": "70%",
            "validation": "15%",
            "test": "15%"
        },
        "preprocessing": {
            "tokenization": "Specyficzna tokenizacja MedicalBERT",
            "lowercase": False,
            "remove_special_characters": True
        },
        "evaluation_metrics": {
            "accuracy": True,
            "f1_score": True,
            "precision": True,
            "recall": True
        },
        "samples": {
            "expected_sample_count": 1500,
            "sample_examples": sample_examples
        }
    }
}

# Zapis do pliku JSON
with open("medicalbert_finetuning_data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("Plik JSON został wygenerowany i zapisany jako 'medicalbert_finetuning_data.json'.")