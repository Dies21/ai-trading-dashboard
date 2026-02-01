"""Тест для перевірки даних dashboard"""
import pandas as pd
from pathlib import Path

# Читаємо CSV
log_file = Path("logs/predictions.csv")
if not log_file.exists():
    print("❌ Файл не знайдено!")
else:
    print(f"✅ Файл знайдено: {log_file}")
    
    df = pd.read_csv(log_file)
    print(f"\n📊 Всього записів: {len(df)}")
    print(f"\n📋 Колонки: {list(df.columns)}")
    
    print(f"\n🔍 Тип колонки timestamp: {df['timestamp'].dtype}")
    print(f"\n📝 Перші 3 значення timestamp:")
    print(df['timestamp'].head(3))
    
    print(f"\n📝 Останні 3 значення timestamp:")
    print(df['timestamp'].tail(3))
    
    # Конвертуємо в datetime
    print(f"\n🔄 Конвертуємо в datetime...")
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    print(f"Тип після конвертації: {df['timestamp'].dtype}")
    print(f"NaT значень: {df['timestamp'].isna().sum()}")
    
    # Останні 10 записів
    latest = df.tail(10).copy()
    print(f"\n📅 Останні 10 timestamp:")
    for idx, ts in enumerate(latest['timestamp'], 1):
        if pd.notna(ts):
            formatted = ts.strftime('%Y-%m-%d %H:%M')
            print(f"  {idx}. {ts} -> {formatted}")
        else:
            print(f"  {idx}. NaT")
