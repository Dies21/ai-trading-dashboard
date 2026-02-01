"""Скрипт для виправлення пошкодженого CSV"""
import pandas as pd
from pathlib import Path

log_file = Path("logs/predictions.csv")
backup_file = Path("logs/predictions_backup.csv")

# Створюємо бекап
import shutil
shutil.copy(log_file, backup_file)
print(f"✅ Створено backup: {backup_file}")

# Читаємо CSV з обробкою помилок
df = pd.read_csv(log_file, on_bad_lines='skip')
print(f"📊 Прочитано {len(df)} записів")

# Перевіряємо колонки
expected_cols = ['timestamp', 'symbol', 'prediction', 'confidence', 'close_price', 
                 'volume', 'balance_simulated', 'p_and_l', 'accuracy', 'win_rate',
                 'horizon', 'resolved', 'actual_direction', 'is_correct']

print(f"\n📋 Очікувані колонки: {len(expected_cols)}")
print(f"📋 Фактичні колонки: {len(df.columns)}")
print(f"📋 Назви колонок: {list(df.columns)}")

# Якщо є зайві колонки, видаляємо їх
if len(df.columns) > len(expected_cols):
    df = df.iloc[:, :len(expected_cols)]
    df.columns = expected_cols
    print(f"\n✂️ Обрізано до {len(expected_cols)} колонок")

# Зберігаємо виправлений файл
df.to_csv(log_file, index=False)
print(f"\n✅ Файл виправлено і збережено")
print(f"\n📝 Останні 3 timestamp:")
print(df['timestamp'].tail(3))
