"""Тестовий скрипт для перевірки resolve_predictions"""
import pandas as pd
from logger import PredictionLogger
from data_loader import CryptoDataLoader

# Створюємо logger
logger = PredictionLogger()

# Завантажуємо дані для декількох символів
loader = CryptoDataLoader()
symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT", "DOT/USDT"]

print("🔍 Перевірка резолюції прогнозів...\n")

for symbol in symbols:
    print(f"\n📊 {symbol}:")
    df = loader.fetch_ohlcv(symbol, timeframe="1h", limit=500)
    
    if df is not None and len(df) > 0:
        print(f"  Завантажено {len(df)} свічок")
        print(f"  Діапазон: {df['time'].min()} - {df['time'].max()}")
        
        # Викликаємо resolve
        resolved = logger.resolve_predictions(symbol, df, horizon=3)
        print(f"  Результат: {resolved} прогнозів розв'язано")
    else:
        print("  ❌ Помилка завантаження даних")

print("\n\n✅ Перевірка завершена. Перегляньте logs/predictions.csv")
