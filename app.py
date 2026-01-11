import streamlit as st
import pandas as pd
import numpy as np
import requests
import asyncio
import aiohttp
import warnings
from datetime import datetime
import plotly.express as px
import nest_asyncio

warnings.filterwarnings('ignore')
nest_asyncio.apply()

month_to_season = {12: 'winter', 1: 'winter', 2: 'winter', 3: 'spring', 4: 'spring', 5: 'spring', 
                   6: 'summer', 7: 'summer', 8: 'summer', 9: 'autumn', 10: 'autumn', 11: 'autumn'}

def process_city_rolling(group):
    group = group.set_index('timestamp')
    group['rolling_30days'] = group['temperature'].rolling('30D', min_periods=1).mean()
    return group.reset_index()

def compute_stats_anomalies(group):
    group['year'] = group['timestamp'].dt.year
    group = group.sort_values('timestamp')
    group['rolling_mean'] = group['temperature'].rolling(window=30, min_periods=1).mean()
    group['season_std'] = group.groupby(['city', 'season'])['temperature'].transform('std').fillna(5)
    group['is_anomaly'] = np.abs(group['temperature'] - group['rolling_mean']) > 2 * group['season_std']
    return group

def get_current_sync(city, apikey):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {'q': city, 'appid': apikey, 'units': 'metric'}
    resp = requests.get(url, params=params)
    if resp.status_code == 200:
        data = resp.json()
        return data['main']['temp']
    if resp.status_code == 401:
        st.error('{"cod":401, "message": "Invalid API key. Please see https://openweathermap.org/faq#error401 for more info."}')
    return None

async def get_current_async(session, city, apikey):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {'q': city, 'appid': apikey, 'units': 'metric'}
    async with session.get(url, params=params) as resp:
        if resp.status == 200:
            data = await resp.json()
            return data['main']['temp']
        return None

async def fetch_all_async(apikey, cities):
    async with aiohttp.ClientSession() as session:
        tasks = [get_current_async(session, city, apikey) for city in cities]
        return await asyncio.gather(*tasks)

st.title("Weather Analysis Dashboard")

uploaded_file = st.file_uploader("Загрузите temperature_data.csv", type="csv")
api_key = st.text_input("OpenWeatherMap API ключ", type="password")

if uploaded_file is not None and api_key:
    df = pd.read_csv(uploaded_file, parse_dates=['timestamp'])
    df['season'] = df['timestamp'].dt.month.map(month_to_season)
    
    selected_city = st.selectbox("Выберите город", sorted(df['city'].unique()))
    city_df = df[df['city'] == selected_city].copy()
    
    df_rolling = process_city_rolling(city_df)
    df_processed = compute_stats_anomalies(df_rolling)
    
    st.header("Описательная статистика")
    desc_stats = df_processed['temperature'].describe().round(2)
    st.write(desc_stats)
    
    season_stats = df_processed.groupby('season')['temperature'].agg(['mean', 'std']).round(2)
    st.dataframe(season_stats)
    
    st.header("Временной ряд с аномалиями")
    fig = px.line(df_processed, x='timestamp', y='temperature', title=f"{selected_city} - температура")
    anomalies = df_processed[df_processed['is_anomaly']]
    fig.add_scatter(x=anomalies['timestamp'], y=anomalies['temperature'],
                    mode='markers', marker=dict(color='red', size=8), name='Аномалии')
    
    st.plotly_chart(fig, width="stretch")

    st.header("Сезонные профили (скользящее среднее)")
    rolling_norms = df_processed.groupby('season')['rolling_mean'].agg(['mean', 'std']).round(2)
    st.dataframe(rolling_norms)
    
    st.header("Текущая температура")


    temps = asyncio.run(fetch_all_async(api_key, [selected_city]))
    current_temp = temps[0]

    if current_temp is None:
        st.error('{"code":401, "message": "Invalid API key. Please see https://openweathermap.org/faq#error401 for more info."}')
    else:
        st.success(f"{current_temp:.1f}°C")


    if current_temp:
        current_month = datetime.now().month
        current_season = month_to_season[current_month]
    
        rolling_norms = df_processed.groupby('season')['rolling_mean'].agg(['mean','std']).round(2)
        norm_mean = rolling_norms.loc[current_season, 'mean']
        norm_std = rolling_norms.loc[current_season, 'std']
    
        is_anomaly = abs(current_temp - norm_mean) > 2 * norm_std
        status = "аномалия" if is_anomaly else "норма"
    
        col1, col2, col3 = st.columns(3)
        col1.metric("Текущая", f"{current_temp:.1f}°C")
        col2.metric("Норма", f"{norm_mean:.1f}°C")
        col3.metric("Диапазон", f"±{norm_std*2:.1f}")
    
        st.info(f"Сезон {current_season}: {status}")

else:
    st.info("Загрузите файл и введите API ключ")
