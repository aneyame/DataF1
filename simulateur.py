import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import streamlit as st

# @st.cache
df = pd.read_csv('merged_all.csv')


# Conversion des temps ( 1:30:28 ) en secondes
def convert_to_seconds(time_str):
    try:
        if pd.isna(time_str) or time_str == '\\N':
            return np.nan
        minutes, rest = time_str.split(':')
        seconds, milliseconds = rest.split('.')
        total_seconds = int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000
        return total_seconds
    except ValueError:
        return np.nan


df['time'] = df['time'].apply(convert_to_seconds)
df['fastestLapTime'] = df['fastestLapTime'].apply(convert_to_seconds)
df.replace('\\N', np.nan, inplace=True)
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
df['fastestLapSpeed'] = df['fastestLapSpeed'].astype(float)

# Sélection des paramètres et de la valeur
x = df[['circuitId', 'Humidity', 'Rainfall', 'TrackTemp', 'AirTemp', 'WindDirection', 'WindSpeed', 'driverId', 'time',
        'fastestLapTime', 'HighTemp', 'fastestLapSpeed']]
y = df[['position']]

# Division des données en jeu d'entraînement et de test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Entrainement du modele ( train et test )
train_data = lgb.Dataset(x_train_scaled, label=y_train)
params = {}
model = lgb.train(params, train_data, num_boost_round=200)

# Style de l'interface
st.title('Simulateur de courses F1 🏁')
st.markdown("""
    <style>
        .main {
            background-color: #000000;
            color: white;
        }
        .reportview-container {
            background-color: #000000;
            color: black;
        }
        .css-1q8dd3e {
            color: white !important;
        }
        .stButton>button {
            background-color: #000000;
            color: black;
        }
    </style>
    """, unsafe_allow_html=True)

circuits = df['circuitRef'].unique()
selected_circuit = st.selectbox('🛤️ Sélectionner un circuit', circuits)

# Choix des paramètres de la course au début
temp = st.slider('🌡️ Température de l\'air (C°)', min_value=int(df['AirTemp'].min()),
                 max_value=int(df['AirTemp'].max()),
                 value=int(df['AirTemp'].mean()))
temp_track = st.slider('🌡️ Température du sol (C°)', min_value=int(df['TrackTemp'].min()),
                       max_value=int(df['TrackTemp'].max()), value=int(df['TrackTemp'].mean()))
humidity = st.slider('💧 Humidité (%)', min_value=int(df['Humidity'].min()), max_value=int(df['Humidity'].max()),
                     value=int(df['Humidity'].mean()))
wind_speed = st.slider('💨 Vitesse du vent', min_value=int(df['WindSpeed'].min()), max_value=int(df['WindSpeed'].max()),
                       value=int(df['WindSpeed'].mean()))
rain = st.radio('🌧️ Présence de pluie?', ('Oui', 'Non'))

high_temp = 1 if temp > 30 else 0
strong_wind = 1 if wind_speed > 20 else 0
pilotes = df['driverRef'].unique()
selected_pilotes = st.multiselect('🏎️ Sélectionner les pilotes', pilotes, default=pilotes[:22])

if len(selected_pilotes) != 22:
    st.warning('⚠️ Vous devez sélectionnez exactement 22 pilotes.')
else:
    circuit_encoded = df[df['circuitRef'] == selected_circuit]['circuitId'].iloc[0]
    results = []

    for idx, driver in enumerate(selected_pilotes):
        driver_encoded = df[df['driverRef'] == driver]['driverId'].iloc[0]

        input_data = [
            [circuit_encoded, humidity, 1 if rain == 'Oui' else 0, temp_track, temp, df['WindDirection'].mean(),
             wind_speed,
             driver_encoded, df['time'].mean(), df['fastestLapTime'].mean(), high_temp, df['fastestLapSpeed'].mean()]]

        if len(input_data[0]) != 12:
            st.error(f"Erreur = le modèle attend 12 caractéristiques, mais {len(input_data[0])} ont été fournies")
        else:
            input_data_scaled = scaler.transform(input_data)
            position_pred = model.predict(input_data_scaled)[0]
            results.append((position_pred, driver))

    results = sorted(results, key=lambda x: x[0])

    # Classement final
    classement_final = []
    for idx, result in enumerate(results):
        classement_final.append([result[1], idx + 1, result[0]])

    # Top 3
    top_3 = classement_final[:3]
    st.subheader("🏆 Top 3 des gagnants de la course")
    for idx, (driver, _, score) in enumerate(top_3):
        st.write(f"**{idx + 1}. {driver}**")

    #Saut de ligne
    st.write("")
    st.write("")
    st.write("")

    # Evaluation du modele
    y_pred = model.predict(x_test_scaled)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)

    # Explication des metriques
    st.subheader('📊 Évaluation du modèle')
    st.write(f"**MAE**: {mae:.2f}")
    st.write(
        "Le **MAE** (Mean Absolute Error) mesure l'erreur moyenne entre les positions prédites et les positions réelles. Plus cette valeur est faible, plus le modèle est précis.\n\n")

    st.write(f"**MSE**: {mse:.2f}")
    st.write(
        "Le **MSE** (Mean Squared Error) est similaire au MAE, mais il pénalise davantage les erreurs importantes. Une valeur faible est aussi un bon indicateur de précision.\n\n")

    ndcg_val = ndcg_score([y_test.values.flatten()], [y_pred.flatten()], k=10)
    st.write(f"**NDCG@10**: {ndcg_val:.2f}")
    st.write(
        "Le **NDCG@10** (Normalized Discounted Cumulative Gain) mesure la qualité du classement des positions prédites, en tenant compte de leur ordre. Une valeur proche de 1 indique un bon classement.\n")
