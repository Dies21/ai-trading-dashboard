"""Тест читання останніх даних для dashboard"""
import pandas as pd
from datetime import datetime

df = pd.read_csv("logs/predictions.csv")
print(f"Всього записів: {len(df)}\n")

latest = df.tail(10).copy()
print("Колонки в CSV:", list(latest.columns))
print("\nПерші 3 значення timestamp (сирі):")
for i in range(min(3, len(latest))):
    print(f"  {i+1}. {latest.iloc[i]['timestamp']}")

# Тепер застосовуємо форматування
def format_timestamp(ts_str):
    if pd.isna(ts_str) or ts_str == '' or ts_str is None:
        return 'N/A'
    try:
        ts_str = str(ts_str).strip()
        if 'T' in ts_str:
            dt = datetime.fromisoformat(ts_str)
        else:
            dt = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S.%f')
        return dt.strftime('%Y-%m-%d %H:%M')
    except Exception as e:
        print(f"    ERROR formatting {ts_str}: {e}")
        return 'N/A'

latest['Час'] = latest['timestamp'].apply(format_timestamp)
print("\nПісля форматування (колонка Час):")
print(latest['Час'].tolist())

print("\nВсі колонки після додавання Час:")
print(list(latest.columns))

# Спробуємо показати DataFrame як у streamlit
display_cols = ['Час', 'symbol', 'prediction', 'confidence', 'close_price']
if 'is_correct' in latest.columns:
    display_cols.append('is_correct')

print(f"\nПоказуємо колонки: {display_cols}")
print(latest[display_cols].head(3))
