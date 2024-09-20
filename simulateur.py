import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import streamlit as st

# @st.cache
df = pd.read_csv('00_testing_clean.csv')


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


x = df[['circuitId', 'Humidity', 'Rainfall', 'TrackTemp', 'AirTemp', 'WindDirection', 'WindSpeed', 'driverId', 'time',
        'fastestLapTime', 'HighTemp', 'fastestLapSpeed']]
y = df[['position']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

train_data = lgb.Dataset(x_train_scaled, label=y_train)
params = {}
model = lgb.train(params, train_data, num_boost_round=200)

st.title('Simulateur de courses F1')

circuits = df['circuitRef'].unique()
selected_circuit = st.selectbox('Selectionner un circuit', circuits)

temp = st.slider('Température de l\'air (C°)', min_value=int(df['AirTemp'].min()), max_value=int(df['AirTemp'].max()),
                 value=int(df['AirTemp'].mean()))
temp_track = st.slider('Température du sol (C°)', min_value=int(df['TrackTemp'].min()),
                       max_value=int(df['TrackTemp'].max()), value=int(df['TrackTemp'].mean()))
humidity = st.slider('Humidité (%)', min_value=int(df['Humidity'].min()), max_value=int(df['Humidity'].max()),
                     value=int(df['Humidity'].mean()))
wind_speed = st.slider('Vitesse du vent', min_value=int(df['WindSpeed'].min()), max_value=int(df['WindSpeed'].max()),
                       value=int(df['WindSpeed'].mean()))
rain = st.radio('Présence de pluie?', ('Oui', 'Non'))
high_temp = 1 if temp > 30 else 0
strong_wind = 1 if wind_speed > 20 else 0
pilotes = df['driverRef'].unique()
selected_pilotes = st.multiselect('Selectionner les pilotes', pilotes, default=pilotes[:22])

if len(selected_pilotes) != 22:
    st.warning('Vous devez sélectionnez exactement 22 pilotes')
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
            results.append((position_pred, driver))  # Correction ici

    # Trier les résultats en fonction de la position prédite (indice 0 du tuple)
    results = sorted(results, key=lambda x: x[0])

    # Construire le classement final
    classement_final = []
    for idx, result in enumerate(results):
        classement_final.append([result[1], idx + 1, result[0]])  # Ajouter le pilote, position estimée et score

    top_3 = classement_final[:3]  # Extraire les trois premiers résultats
    st.subheader("Top 3 des gagnants de la course")
    for idx, (driver, _, score) in enumerate(top_3):
        st.write(f"{idx + 1}. {driver}")  # Utiliser l'index pour le classement

    y_pred = model.predict(x_test_scaled)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    st.write(f"MAE: {mae:.2f}")
    st.write(f"MSE: {mse:.2f}")

    ndcg_val = ndcg_score([y_test.values.flatten()], [y_pred.flatten()], k=10)
    st.write(f"NDCG@10: {ndcg_val:.2f}")








