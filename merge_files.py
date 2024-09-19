import pandas as pd

file_path = 'F1 Weather(2023-2018) 1.csv'
weather_data = pd.read_csv(file_path)
weather_data.head()

weather_data = pd.read_csv(file_path, sep=';')
weather_data.head()

averaged_data = weather_data.groupby(['name', 'Year']).mean(numeric_only=True).reset_index()
print(averaged_data.head())

averaged_data.to_csv('averaged_weather_data_by_track_and_year.csv', index=False)