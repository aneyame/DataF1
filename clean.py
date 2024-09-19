import pandas as pd

weather_data = pd.read_csv('merged_all 1.csv', sep=';')
print(weather_data.columns)

cleaned_data = weather_data.drop_duplicates(subset=['name', 'Year', 'AirTemp', 'Humidity', 'Pressure', 'Rainfall', 'TrackTemp', 'WindDirection', 'WindSpeed', 'Round Number', 'raceId', 'round', 'circuitId', 'date', 'time', 'driverStandingsId', 'driverId', 'points', 'wins', 'driverRef'])

cleaned_data.to_csv('00_testing_clean.csv', index=False)
weather_data.to_csv('merged_all_comma.csv', index=False)

print(weather_data.columns)
print(cleaned_data.head())