"""Детальна діагностика проблеми з timestamp"""
import pandas as pd

df = pd.read_csv("logs/predictions.csv")
print(f"Всього: {len(df)} записів\n")

# Подивимося на сирі значення останніх 10
print("📝 Останні 10 СИРИХ значень timestamp:")
for idx in range(-10, 0):
    row_idx = len(df) + idx
    ts_raw = df.iloc[idx]['timestamp']
    print(f"  Рядок {row_idx}: repr={repr(ts_raw)}, type={type(ts_raw)}, len={len(str(ts_raw))}")

# Спробуємо конвертувати вручну
print("\n🔄 Ручна конвертація останніх 10:")
for idx in range(-10, 0):
    ts_raw = df.iloc[idx]['timestamp']
    try:
        ts_converted = pd.to_datetime(ts_raw)
        print(f"  ✅ {ts_raw} -> {ts_converted}")
    except Exception as e:
        print(f"  ❌ {ts_raw} -> ERROR: {e}")

# Перевіримо, можливо є невидимі символи
print("\n🔍 Перевірка на невидимі символи:")
ts = df.iloc[-1]['timestamp']
print(f"Останній timestamp: {repr(ts)}")
print(f"Bytes: {ts.encode('utf-8') if isinstance(ts, str) else 'not string'}")
