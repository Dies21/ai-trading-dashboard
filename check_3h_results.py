import pandas as pd
from datetime import datetime, timedelta

df = pd.read_csv('logs/predictions.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

# Прогнозы от примерно 3 часа назад (12:41)
three_hours_ago = datetime.now() - timedelta(hours=3)
old_preds = df[df['timestamp'] < three_hours_ago]

print(f"Текущее время: {datetime.now().strftime('%H:%M:%S')}")
print(f"3 часа назад: {three_hours_ago.strftime('%H:%M:%S')}")
print()
print(f"Прогнозов старше 3 часов: {len(old_preds)}")
print()

# Статус проверки
resolved = old_preds[~old_preds['is_correct'].isna()]
unresolved = old_preds[old_preds['is_correct'].isna()]

print(f"✅ Проверено: {len(resolved)}")
print(f"⏳ Не проверено: {len(unresolved)}")
print()

if len(resolved) > 0:
    correct = (resolved['is_correct'] == 'TRUE').sum()
    print(f"Из проверенных: {correct}/{len(resolved)} верных = {100*correct/len(resolved):.1f}%")
    print()
    print("Примеры проверенных:")
    print(resolved[['timestamp', 'symbol', 'prediction', 'is_correct', 'price_change']].head(10).to_string())
