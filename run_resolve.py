"""Скрипт для ручного запуска resolve_predictions для всех символов."""
import pandas as pd
from pathlib import Path
from logger import PredictionLogger

def main():
    logger = PredictionLogger()
    candles_dir = Path("logs/candles")
    
    if not candles_dir.exists():
        print("❌ Папка logs/candles не найдена")
        return
    
    symbols_resolved = 0
    total_resolved = 0
    
    # Получаем список всех символов из CSV
    csv_path = Path("logs/predictions.csv")
    if not csv_path.exists():
        print("❌ Файл predictions.csv не найден")
        return
    
    df = pd.read_csv(csv_path)
    unique_symbols = df['symbol'].unique()
    
    print(f"📊 Найдено {len(unique_symbols)} символов для проверки\n")
    
    for symbol in unique_symbols:
        # Конвертируем BTC/USDT -> BTC_USDT для имени файла
        file_symbol = symbol.replace('/', '_')
        candle_file = candles_dir / f"{file_symbol}.parquet"
        
        if not candle_file.exists():
            print(f"⚠️ {symbol}: нет файла свечей {candle_file.name}")
            continue
        
        try:
            # Загружаем свечи
            candles = pd.read_parquet(candle_file)
            print(f"🔍 {symbol}: загружено {len(candles)} свечей")
            
            # Запускаем resolve
            resolved = logger.resolve_predictions(symbol, candles, horizon=3)
            
            if resolved > 0:
                symbols_resolved += 1
                total_resolved += resolved
                print(f"   ✅ Разрешено {resolved} прогнозов\n")
            else:
                print(f"   ⏳ Нет прогнозов для разрешения\n")
                
        except Exception as e:
            print(f"❌ {symbol}: ошибка - {e}\n")
    
    print(f"\n{'='*60}")
    print(f"📊 ИТОГО:")
    print(f"   Символов обработано: {symbols_resolved}/{len(unique_symbols)}")
    print(f"   Прогнозов разрешено: {total_resolved}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
