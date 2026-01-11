import pandas as pd
import numpy as np
import time
from joblib import Parallel, delayed
import warnings
import requests
import asyncio
import aiohttp
import pandas as pd
import nest_asyncio
from datetime import datetime

nest_asyncio.apply()
warnings.filterwarnings('ignore')

np.random.seed(42)

seasonal_temperatures = {
    "New York": {"winter": 0, "spring": 10, "summer": 25, "autumn": 15},
    "London": {"winter": 5, "spring": 11, "summer": 18, "autumn": 12},
    "Paris": {"winter": 4, "spring": 12, "summer": 20, "autumn": 13},
    "Tokyo": {"winter": 6, "spring": 15, "summer": 27, "autumn": 18},
    "Moscow": {"winter": -10, "spring": 5, "summer": 18, "autumn": 8},
    "Sydney": {"winter": 12, "spring": 18, "summer": 25, "autumn": 20},
    "Berlin": {"winter": 0, "spring": 10, "summer": 20, "autumn": 11},
    "Beijing": {"winter": -2, "spring": 13, "summer": 27, "autumn": 16},
    "Rio de Janeiro": {"winter": 20, "spring": 25, "summer": 30, "autumn": 25},
    "Dubai": {"winter": 20, "spring": 30, "summer": 40, "autumn": 30},
    "Los Angeles": {"winter": 15, "spring": 18, "summer": 25, "autumn": 20},
    "Singapore": {"winter": 27, "spring": 28, "summer": 28, "autumn": 27},
    "Mumbai": {"winter": 25, "spring": 30, "summer": 35, "autumn": 30},
    "Cairo": {"winter": 15, "spring": 25, "summer": 35, "autumn": 25},
    "Mexico City": {"winter": 12, "spring": 18, "summer": 20, "autumn": 15},
}

month_to_season = {12: "winter", 1: "winter", 2: "winter",
                   3: "spring", 4: "spring", 5: "spring",
                   6: "summer", 7: "summer", 8: "summer",
                   9: "autumn", 10: "autumn", 11: "autumn"}

def generate_realistic_temperature_data(cities, num_years=10):
    dates = pd.date_range(start="2010-01-01", periods=365 * num_years, freq="D")
    data = []
    for city in cities:
        for date in dates:
            season = month_to_season[date.month]
            mean_temp = seasonal_temperatures[city][season]
            temperature = np.random.normal(loc=mean_temp, scale=5)
            data.append({"city": city, "timestamp": date, "temperature": temperature})
    df = pd.DataFrame(data)
    df['season'] = df['timestamp'].dt.month.map(month_to_season)
    return df

data = generate_realistic_temperature_data(list(seasonal_temperatures.keys()))
data.to_csv('temperature_data.csv', index=False)
df = pd.read_csv('temperature_data.csv', parse_dates=['timestamp'])
print(f"Данные загружены: {df.shape}")

def process_city_rolling(group):
    """Скользящее среднее для одного города"""
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


print("\nПоследовательный анализ")
start_seq = time.time()

# Скользящее среднее
df_seq = df.groupby('city', group_keys=False).apply(process_city_rolling)

# Статистики и аномалии
df_seq = df_seq.groupby('city', group_keys=False).apply(compute_stats_anomalies)

# Сезонные статистики (общие по сезонам, без года)
season_stats_seq = df_seq.groupby(['city', 'season'])['temperature'].agg(['mean', 'std']).round(2)
stats_seq = df_seq.groupby(['city', 'season', 'year'])['temperature'].agg(['mean', 'std', 'count']).round(2)

time_seq = time.time() - start_seq
print(f"Время последовательного анализа: {time_seq:.3f}с")
print(f"Аномалий: {df_seq['is_anomaly'].sum()} ({df_seq['is_anomaly'].mean():.1%})")

print("\nПараллельный анализ")
start_par = time.time()

# Скользящее среднее параллельно (по городам)
results_rolling = Parallel(n_jobs=-1)(delayed(process_city_rolling)(group) for name, group in df.groupby('city'))
df_par = pd.concat(results_rolling)

# Статистики и аномалии параллельно
results_stats = Parallel(n_jobs=-1)(delayed(compute_stats_anomalies)(group) for name, group in df_par.groupby('city'))
df_par = pd.concat(results_stats)

season_stats_par = df_par.groupby(['city', 'season'])['temperature'].agg(['mean', 'std']).round(2)
stats_par = df_par.groupby(['city', 'season', 'year'])['temperature'].agg(['mean', 'std', 'count']).round(2)

time_par = time.time() - start_par
print(f"Время параллельного анализа: {time_par:.3f}с")
print(f"Ускорение: {time_seq/time_par:.1f}x")
print(f"Аномалий: {df_par['is_anomaly'].sum()}")

print("\nПроверка, что данные совпадают:")
assert df_seq['rolling_30days'].equals(df_par['rolling_30days']), "Rolling не совпадает"
assert df_seq['is_anomaly'].sum() == df_par['is_anomaly'].sum(), "Аномалии не совпадают"
print("Результаты идентичны")

# Прироста производительности нет, так как в данном случае больше времени тратится на выделение памяти и создание процессов, при большем датасете
# и более сложных вычислениях был бы выигрыш

api_key = 'your_api_key'
cities = ['Berlin', 'Cairo', 'Dubai', 'Beijing', 'Moscow']
url = 'https://api.openweathermap.org/data/2.5/weather'

df_hist = pd.read_csv('temperature_data.csv', parse_dates=['timestamp'])
df_hist = df_hist.sort_values(['city', 'timestamp'])
df_hist['rolling_mean'] = df_hist.groupby('city')['temperature'].rolling(window=30, min_periods=1).mean().reset_index(0, drop=True)
df_hist['season'] = df_hist['timestamp'].dt.month.map(month_to_season)
season_norms = df_hist.groupby(['city', 'season'])[['rolling_mean']].agg(['mean', 'std']).round(2)
current_season = month_to_season[datetime.now().month]

# Синхронно
def get_current_sync(city):
    params = {'q': city, 'appid': api_key, 'units': 'metric'}
    resp = requests.get(url, params=params)
    if resp.status_code == 200:
        data = resp.json()
        return {'city': city, 'temp': data['main']['temp'], 'season': current_season}
    return {'city': city, 'temp': None, 'season': current_season, 'error': resp.status_code}

print('Синхронные запросы')
sync_results = []
start_sync = pd.Timestamp.now()
for city in cities:
    result = get_current_sync(city)
    sync_results.append(result)
    print(f"{result['city']}: {result['temp']:.1f}°C" if result['temp'] else f"{result['city']}: ERROR")
time_sync = (pd.Timestamp.now() - start_sync).total_seconds()
print(f'Время синхронно: {time_sync:.2f}с\n')


# Асинхронно
async def get_current_async(session, city):
    params = {'q': city, 'appid': api_key, 'units': 'metric'}
    async with session.get(url, params=params) as resp:
        if resp.status == 200:
            data = await resp.json()
            return {'city': city, 'temp': data['main']['temp'], 'season': current_season}
        return {'city': city, 'temp': None, 'season': current_season, 'error': resp.status}

async def fetch_all_async():
    async with aiohttp.ClientSession() as session:
        tasks = [get_current_async(session, city) for city in cities]
        return await asyncio.gather(*tasks)

print('Асинхронные запросы')
start_async = pd.Timestamp.now()
async_results = asyncio.run(fetch_all_async())
for res in async_results:
    print(f"{res['city']}: {res['temp']:.1f}°C" if res['temp'] else f"{res['city']}: ERROR")
time_async = (pd.Timestamp.now() - start_async).total_seconds()
print(f'Время асинхронно: {time_async:.2f}с\n')

all_results = sync_results + [dict(res) for res in async_results]
print('Аномалии')
for row in all_results:
    if row['temp'] is not None and row['city'] in season_norms.index.get_level_values('city'):
        try:
            norm = season_norms.loc[(row['city'], current_season), 'rolling_mean']
            is_anomaly = abs(row['temp'] - norm['mean']) > 2 * norm['std']
            status = 'Аномалия' if is_anomaly else 'Норма'
            print(f"{row['city']}: {row['temp']:.1f}°C ; норма {norm['mean']:.1f} +/- {2*norm['std']:.1f}°C : {status}")
        except KeyError:
            print(f"{row['city']}: Нет норм")

# Асинхронность показала очень хороший прирост: примерно x5. Однозначно в этом случае лучше использовать асинхронные методы.
