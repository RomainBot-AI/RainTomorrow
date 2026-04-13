import streamlit as st
import pandas as pd
import numpy as np
import json
from xgboost import XGBClassifier

st.set_page_config(page_title="Rain Prediction App", page_icon="🌧️", layout="centered")

MODEL_JSON_PATH = "model.json"
METADATA_PATH = "app_metadata.json"
IMPUTER_VALUES_PATH = "imputer_values.json"

AUSTRALIAN_LOCATIONS = [
    "Perth",
    "Sydney",
    "Melbourne",
    "Brisbane",
    "Adelaide",
    "Darwin",
    "Hobart",
    "Canberra",
]


@st.cache_resource
def load_artifacts():
    model = XGBClassifier()
    model.load_model(MODEL_JSON_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    with open(IMPUTER_VALUES_PATH, "r", encoding="utf-8") as f:
        imputer_values = np.array(json.load(f), dtype=float)

    return model, metadata, imputer_values


def clean_column_name(col: str) -> str:
    col = str(col).strip().lower()
    col = col.replace("�", "°")
    col = " ".join(col.split())
    return col


def read_bom_csv(uploaded_file) -> pd.DataFrame:
    """
    Lit de façon robuste un CSV BoM dont la ligne d'en-tête
    peut varier selon les fichiers.
    """
    encodings = ["utf-8", "latin-1"]
    candidate_skiprows = range(0, 15)

    for encoding in encodings:
        for skip in candidate_skiprows:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=encoding, skiprows=skip)

                if df.shape[1] == 0:
                    continue

                # Supprime la première colonne vide éventuelle
                first_col = str(df.columns[0]).strip()
                if first_col.startswith("Unnamed") or first_col == "":
                    df = df.iloc[:, 1:]

                cleaned_cols = [clean_column_name(c) for c in df.columns]

                # Heuristique : si on trouve "date" et quelques colonnes météo,
                # on considère qu'on a trouvé le bon header
                if "date" in cleaned_cols and (
                    "minimum temperature (°c)" in cleaned_cols
                    or "maximum temperature (°c)" in cleaned_cols
                    or "rainfall (mm)" in cleaned_cols
                ):
                    return df

            except Exception:
                continue

    raise ValueError(
        "Impossible de trouver automatiquement la ligne d'en-tête du CSV BoM."
    )


def normalize_bom_dataframe(df: pd.DataFrame, location: str) -> pd.DataFrame:
    df = df.copy()

    # Nettoyage robuste des noms de colonnes
    cleaned_original_cols = {col: clean_column_name(col) for col in df.columns}
    df = df.rename(columns=cleaned_original_cols)

    rename_map = {
        "date": "Date",
        "minimum temperature (°c)": "MinTemp",
        "maximum temperature (°c)": "MaxTemp",
        "rainfall (mm)": "Rainfall",
        "evaporation (mm)": "Evaporation",
        "sunshine (hours)": "Sunshine",
        "direction of maximum wind gust": "WindGustDir",
        "speed of maximum wind gust (km/h)": "WindGustSpeed",
        "time of maximum wind gust": "WindGustTime",
        "9am temperature (°c)": "Temp9am",
        "9am relative humidity (%)": "Humidity9am",
        "9am cloud amount (oktas)": "Cloud9am",
        "9am wind direction": "WindDir9am",
        "9am wind speed (km/h)": "WindSpeed9am",
        "9am msl pressure (hpa)": "Pressure9am",
        "3pm temperature (°c)": "Temp3pm",
        "3pm relative humidity (%)": "Humidity3pm",
        "3pm cloud amount (oktas)": "Cloud3pm",
        "3pm wind direction": "WindDir3pm",
        "3pm wind speed (km/h)": "WindSpeed3pm",
        "3pm msl pressure (hpa)": "Pressure3pm",
    }

    df = df.rename(columns=lambda x: rename_map.get(x, x))

    # Ajout de la localisation choisie
    df["Location"] = location

    # Vérification Date
    if "Date" not in df.columns:
        raise ValueError(
            f"Colonne 'Date' introuvable après normalisation. Colonnes disponibles : {list(df.columns)}"
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    numeric_cols = [
        "MinTemp",
        "MaxTemp",
        "Rainfall",
        "Evaporation",
        "Sunshine",
        "WindGustSpeed",
        "Temp9am",
        "Humidity9am",
        "Cloud9am",
        "WindSpeed9am",
        "Pressure9am",
        "Temp3pm",
        "Humidity3pm",
        "Cloud3pm",
        "WindSpeed3pm",
        "Pressure3pm",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Rainfall" in df.columns:
        df["RainToday"] = df["Rainfall"].apply(
            lambda x: "Yes" if pd.notna(x) and x > 1.0 else "No"
        )
    else:
        df["RainToday"] = np.nan

    return df


def add_engineered_features_for_app(
    X: pd.DataFrame,
    rainfall_median: float,
    pressure_median: float
) -> pd.DataFrame:
    X = X.copy()

    if {"MaxTemp", "MinTemp"}.issubset(X.columns):
        X["TempRange"] = X["MaxTemp"] - X["MinTemp"]

    if {"Humidity3pm", "Humidity9am"}.issubset(X.columns):
        X["HumidityChange"] = X["Humidity3pm"] - X["Humidity9am"]

    if {"Pressure3pm", "Pressure9am"}.issubset(X.columns):
        X["PressureChange"] = X["Pressure3pm"] - X["Pressure9am"]

    if {"Humidity3pm", "Cloud3pm"}.issubset(X.columns):
        X["HumidityCloudInteraction"] = X["Humidity3pm"] * X["Cloud3pm"]

    if {"MaxTemp", "Humidity3pm"}.issubset(X.columns):
        X["TempHumidityInteraction"] = X["MaxTemp"] * X["Humidity3pm"]

    if "Humidity3pm" in X.columns:
        X["HighHumidity3pm"] = (X["Humidity3pm"] >= 70).astype(int)

    if "Pressure3pm" in X.columns:
        X["LowPressure3pm"] = (X["Pressure3pm"] <= pressure_median).astype(int)

    if "Rainfall" in X.columns:
        X["RainfallAboveMedian"] = (X["Rainfall"] > rainfall_median).astype(int)

    return X


def build_model_input_from_row(row: pd.Series, metadata: dict) -> pd.DataFrame:
    available_num_cols = metadata["available_num_cols"]
    available_cat_cols = metadata["available_cat_cols"]
    base_feature_cols = metadata["base_feature_cols"]
    feature_cols = metadata["feature_cols"]
    category_levels = metadata["category_levels"]
    rainfall_median = metadata["rainfall_median_train"]
    pressure_median = metadata["pressure_median_train"]

    base_data = {}

    for col in available_num_cols:
        base_data[col] = row.get(col, np.nan)

    for col in available_cat_cols:
        value = row.get(col, np.nan)
        if pd.notna(value):
            value = str(value)
        base_data[col] = value

    base_df = pd.DataFrame([base_data])
    ohe_df = base_df.copy()

    for cat_col in available_cat_cols:
        levels = category_levels.get(cat_col, [])
        value = str(base_df.loc[0, cat_col]) if pd.notna(base_df.loc[0, cat_col]) else None

        # drop_first=True à l'entraînement
        for level in levels[1:]:
            col_name = f"{cat_col}_{level}"
            ohe_df[col_name] = 1 if value == level else 0

    ohe_df = ohe_df.drop(columns=available_cat_cols, errors="ignore")

    for col in base_feature_cols:
        if col not in ohe_df.columns:
            ohe_df[col] = 0

    ohe_df = ohe_df[base_feature_cols]

    final_df = add_engineered_features_for_app(
        ohe_df,
        rainfall_median=rainfall_median,
        pressure_median=pressure_median
    )

    for col in feature_cols:
        if col not in final_df.columns:
            final_df[col] = 0.0

    final_df = final_df[feature_cols].astype(float)
    return final_df


def manual_impute(model_input: pd.DataFrame, imputer_values: np.ndarray) -> np.ndarray:
    model_input_np = model_input.to_numpy(dtype=float)
    mask = np.isnan(model_input_np)

    if mask.any():
        row_idx, col_idx = np.where(mask)
        model_input_np[row_idx, col_idx] = imputer_values[col_idx]

    return model_input_np


st.title("Rain Prediction App")
st.write(
    "Charge un fichier CSV BoM, choisis une localisation et une date, puis prédis la pluie du lendemain."
)
st.info("""
📥 **Comment récupérer les données météo ?**

Les données proviennent du site officiel du Bureau of Meteorology :

👉 https://www.bom.gov.au/climate/dwo/index.shtml

**Étapes :**

1. Clique sur une ville australienne (ex : Sydney, Perth…)
2. Sélectionne un mois dans la liste (ex : Apr 26, Mar 26…)
3. Une fois sur la page du mois :
   - utilise **"Other formats"**
   - télécharge le fichier en **CSV (plain text version)**

4. Importe ce fichier dans l'application

""")

try:
    model, metadata, imputer_values = load_artifacts()
except Exception as e:
    st.error(f"Impossible de charger les artefacts : {e}")
    st.stop()

uploaded_file = st.file_uploader("Importer un CSV BoM", type=["csv"])
selected_location = st.selectbox("Choisir une localisation", AUSTRALIAN_LOCATIONS)

if uploaded_file is not None:
    try:
        raw_df = read_bom_csv(uploaded_file)
        df = normalize_bom_dataframe(raw_df, selected_location)

        st.subheader("Aperçu du CSV normalisé")
        st.dataframe(df.head(), use_container_width=True)

        valid_dates = df["Date"].dropna().dt.date.sort_values().unique()

        if len(valid_dates) == 0:
            st.error("Aucune date valide n'a été trouvée dans le CSV.")
            st.stop()

        selected_date = st.selectbox("Choisir une date disponible", valid_dates)

        filtered_df = df[df["Date"].dt.date == selected_date]

        if filtered_df.empty:
            st.warning("Aucune ligne trouvée pour cette date.")
            st.stop()

        row = filtered_df.iloc[0]

        st.subheader("Données utilisées")
        st.json({
            "Location": row.get("Location"),
            "Date": row.get("Date").date() if pd.notna(row.get("Date")) else None,
            "MinTemp": row.get("MinTemp"),
            "MaxTemp": row.get("MaxTemp"),
            "Rainfall": row.get("Rainfall"),
            "Evaporation": row.get("Evaporation"),
            "Sunshine": row.get("Sunshine"),
            "WindGustDir": row.get("WindGustDir"),
            "WindGustSpeed": row.get("WindGustSpeed"),
            "WindDir9am": row.get("WindDir9am"),
            "WindDir3pm": row.get("WindDir3pm"),
            "WindSpeed9am": row.get("WindSpeed9am"),
            "WindSpeed3pm": row.get("WindSpeed3pm"),
            "Humidity9am": row.get("Humidity9am"),
            "Humidity3pm": row.get("Humidity3pm"),
            "Pressure9am": row.get("Pressure9am"),
            "Pressure3pm": row.get("Pressure3pm"),
            "Cloud9am": row.get("Cloud9am"),
            "Cloud3pm": row.get("Cloud3pm"),
            "Temp9am": row.get("Temp9am"),
            "Temp3pm": row.get("Temp3pm"),
            "RainToday": row.get("RainToday"),
        })

        st.subheader("Résumé")
        c1, c2, c3 = st.columns(3)
        c1.metric("MinTemp", f"{row.get('MinTemp', 'N/A')} °C")
        c2.metric("MaxTemp", f"{row.get('MaxTemp', 'N/A')} °C")
        c3.metric("Rainfall", f"{row.get('Rainfall', 'N/A')} mm")

        c4, c5, c6 = st.columns(3)
        c4.metric("Humidity 3pm", f"{row.get('Humidity3pm', 'N/A')} %")
        c5.metric("Pressure 3pm", f"{row.get('Pressure3pm', 'N/A')} hPa")
        c6.metric("Wind 3pm", f"{row.get('WindSpeed3pm', 'N/A')} km/h")

        if st.button("Prédire la pluie demain"):
            model_input = build_model_input_from_row(row, metadata)
            model_input_imputed = manual_impute(model_input, imputer_values)

            proba = float(model.predict_proba(model_input_imputed)[0, 1])
            pred_label = "Pluie" if proba >= 0.5 else "Pas de pluie"

            st.subheader("Résultat")
            r1, r2 = st.columns(2)
            r1.metric("Prédiction", pred_label)
            r2.metric("Probabilité de pluie", f"{proba * 100:.1f}%")

            if proba < 0.40:
                st.success("Risque faible de pluie demain.")
            elif proba < 0.60:
                st.warning("Risque intermédiaire de pluie demain.")
            else:
                st.error("Risque élevé de pluie demain.")

            with st.expander("Voir les features envoyées au modèle"):
                st.dataframe(model_input, use_container_width=True)

    except Exception as e:
        st.error(f"Erreur lors du chargement ou de la prédiction : {e}")
else:
    st.info("Importe un CSV BoM pour commencer.")
