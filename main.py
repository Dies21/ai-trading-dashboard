import time
import subprocess
from datetime import datetime
from data_loader import CryptoDataLoader
from features import add_indicators
from patterns import detect_all_patterns
from model import train_model, evaluate_model, analyze_feature_importance
from predictor import predict_next
from logger import PredictionLogger

def auto_push_logs():
    """Автоматически отправляет обновленные логи в GitHub"""
    try:
        print("\n📤 Отправка данных на сайт...")
        subprocess.run(["git", "add", "-f", "logs/predictions.csv"], check=False, capture_output=True)
        commit_msg = f"auto-update: predictions {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        result = subprocess.run(["git", "commit", "-m", commit_msg], check=False, capture_output=True, text=True)
        
        if "nothing to commit" not in result.stdout:
            subprocess.run(["git", "push", "origin", "main"], check=False, capture_output=True)
            print("✅ Данные успешно отправлены на сайт")
        else:
            print("ℹ️ Нет новых данных для отправки")
    except Exception as e:
        print(f"⚠️ Не удалось отправить данные: {e}")

if __name__ == "__main__":
    # Инициализация
    loader = CryptoDataLoader(symbols=[
        "BTC/USDT", 
        "ETH/USDT",
        "BNB/USDT",
        "XRP/USDT",
        "SOL/USDT",
        "ADA/USDT",
        "DOGE/USDT",
        "MATIC/USDT",
        "DOT/USDT",
        "AVAX/USDT",
        "PEPE/USDT",
        "SUI/USDT",
        "ENA/USDT",
        "LTC/USDT",
        "LINK/USDT"
    ])
    logger = PredictionLogger(log_dir="logs")
    
    # Настройки эффективности обучения
    TRAIN_INTERVAL_HOURS = 6
    MAX_TRAIN_ROWS = 1500
    model_cache = {}
    last_train_time = {}
    
    iteration = 0
    total_predictions = 0
    profitable_predictions = 0

    while True:
        iteration += 1
        print(f"\n{'='*70}")
        print(f"Итерация {iteration} | {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")
        
        try:
            # Загрузить данные для всех активов
            data_dict = loader.fetch_multiple(timeframe="1h", limit=500)
            
            if not data_dict:
                print("Ошибка: не удалось загрузить данные")
                time.sleep(60)
                continue
            
            # Обработать каждый актив
            for symbol, df in data_dict.items():
                print(f"\n📊 Анализирую {symbol}...")
                print("-" * 70)
                
                # ПРОВЕРКА СТАРЫХ ПРОГНОЗОВ (resolve)
                try:
                    resolved_count = logger.resolve_predictions(symbol, df, horizon=1)
                    if resolved_count > 0:
                        print(f"   ✅ Разрешено {resolved_count} старых прогнозов")
                except Exception as e:
                    print(f"   ⚠️ Ошибка при resolve: {e}")
                
                df = add_indicators(df)
                df = detect_all_patterns(df)

                # Ограничиваем объем данных для обучения
                df_train = df.tail(MAX_TRAIN_ROWS)

                # Переобучаем модель не чаще чем раз в N часов
                now = datetime.now()
                need_train = (
                    symbol not in model_cache or
                    symbol not in last_train_time or
                    (now - last_train_time[symbol]).total_seconds() >= TRAIN_INTERVAL_HOURS * 3600
                )

                # Обучить модель только если нужно
                if need_train:
                    model = train_model(df_train)
                    model_cache[symbol] = model
                    last_train_time[symbol] = now
                else:
                    model = model_cache[symbol]
                
                # Анализ важности признаков
                analyze_feature_importance(model, top_n=3)
                
                # Оценка модели
                metrics = evaluate_model(model, df_train)

                # Предсказание с агрессивными DOWN трешолдами и учётом паттернов
                prediction, confidence, prob_down, prob_up, reliability, pattern_up, pattern_down = predict_next(model, df, up_threshold=0.48, down_threshold=0.35)
                
                # Красивый вывод предсказания
                if prediction == "UP":
                    emoji = "⬆ Вверх"
                elif prediction == "DOWN":
                    emoji = "⬇ Вниз"
                elif prediction == "UNSURE":
                    emoji = "❓ Неуверен"
                else:
                    emoji = "⚠ Нет данных"
                
                # Відображення надійності
                if reliability == "HIGH":
                    rel_emoji = "🟢 Висока"
                elif reliability == "MEDIUM":
                    rel_emoji = "🟡 Середня"
                elif reliability == "LOW":
                    rel_emoji = "🔴 Низька"
                else:
                    rel_emoji = "⚪ Невідома"
                
                print(f"\n🎯 Прогноз: {emoji}")
                print(f"   Уверенность: {confidence:.2%} (DOWN: {prob_down:.2%}, UP: {prob_up:.2%})")
                print(f"   Паттерны: 🔴 DOWN={pattern_down} | 🟢 UP={pattern_up}")
                print(f"   Надёжность: {rel_emoji}")
                
                # Данные для логирования
                close_price = df["close"].iloc[-1]
                volume = df["volume"].iloc[-1]
                candle_time = df["time"].iloc[-1] if "time" in df.columns else None
                
                # Получить метрики из evaluate_model (это словарь)
                accuracy = metrics.get("accuracy", 0)
                win_rate = metrics.get("f1", 0) * 100  # Приблизительная оценка
                
                # Симулированная торговля (простая версия)
                balance_change = 100 * accuracy - 50  # Упрощённая формула
                
                # Логировать предсказание
                log_entry = logger.log_prediction(
                    symbol=symbol,
                    prediction=prediction,
                    confidence=confidence,
                    close_price=close_price,
                    volume=volume,
                    balance_simulated=1000 + balance_change,
                    p_and_l=balance_change,
                    accuracy=accuracy,
                    win_rate=win_rate,
                    timestamp=candle_time
                )
                
                print(f"\n💾 Данные логированы:")
                print(f"   Цена: ${close_price:.2f}")
                print(f"   Объём: {volume:.0f}")
                print(f"   Точность: {accuracy:.2%}")
                
                # Статистика
                total_predictions += 1
                if prediction in ["UP", "DOWN"] and confidence > 0.6:
                    profitable_predictions += 1
            
            # Итоги итерации
            print(f"\n{'='*70}")
            print(f"ИТОГИ ИТЕРАЦИИ #{iteration}")
            print(f"Активов проанализировано: {len(data_dict)}")
            print(f"Всего предсказаний: {total_predictions}")
            print(f"Уверенных предсказаний: {profitable_predictions}")
            
            # Статистика
            stats = logger.get_statistics()
            if stats:
                print(f"\n📈 Статистика:")
                print(f"   Всего записей: {stats['total_predictions']}")
                print(f"   Активов: {stats['symbols']}")
                print(f"   Средняя уверенность: {stats['avg_confidence']:.2%}")
                print(f"   UP: {stats['up_predictions']} | DOWN: {stats['down_predictions']} | UNSURE: {stats['unsure_predictions']}")
            
            # Автоматическая отправка логов на сайт
            auto_push_logs()
            
            print(f"\n⏱️  Следующая итерация через 1800 сек (30 минут)...\n")
            
        except Exception as e:
            print(f"❌ Ошибка: {e}\n")
            import traceback
            traceback.print_exc()
        
        # Ждём 30 минут перед следующей итерацией
        time.sleep(1800)
