import pandas as pd
import json
from datetime import datetime
from pathlib import Path

class PredictionLogger:
    """Система логирования предсказаний модели."""
    
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # CSV лог для быстрого анализа
        self.csv_log = self.log_dir / "predictions.csv"
        self.ensure_csv_header()
        self.ensure_csv_columns()
    
    def ensure_csv_header(self):
        """Создать заголовок CSV если файл не существует."""
        if not self.csv_log.exists():
            header = pd.DataFrame(columns=self._expected_columns())
            header.to_csv(self.csv_log, index=False)

    def _expected_columns(self):
        return [
            'timestamp',
            'symbol',
            'prediction',
            'confidence',
            'close_price',
            'exit_price',
            'price_change_pct',
            'price_change_abs',
            'volume',
            'balance_simulated',
            'p_and_l',
            'accuracy',
            'win_rate',
            'horizon',
            'resolved',
            'actual_direction',
            'is_correct'
        ]

    def ensure_csv_columns(self):
        """Ensure CSV has all expected columns (upgrade old schema if needed)."""
        if not self.csv_log.exists():
            return
        try:
            header = pd.read_csv(self.csv_log, nrows=0)
            expected = self._expected_columns()
            missing = [c for c in expected if c not in header.columns]
            if not missing:
                return
            df = pd.read_csv(self.csv_log, on_bad_lines='skip', engine='python')
            for col in missing:
                df[col] = ""
            df = df.reindex(columns=expected)
            df.to_csv(self.csv_log, index=False)
        except Exception:
            return
    
    def log_prediction(self, symbol, prediction, confidence, close_price, volume, 
                      balance_simulated, p_and_l, accuracy, win_rate, horizon=1, timestamp=None):
        """Логировать одно предсказание.
        
        Аргументы:
            symbol: название актива (BTC/USD, ETH/USD и т.д.)
            prediction: предсказание (UP, DOWN, UNSURE)
            confidence: уверенность модели (0-1)
            close_price: цена закрытия
            volume: объём
            balance_simulated: симулированный баланс
            p_and_l: прибыль/убыток
            accuracy: точность на тестовой выборке
            win_rate: процент прибыльных сделок
        """
        ts = timestamp if timestamp is not None else datetime.now().isoformat()
        log_data = {
            'timestamp': ts,
            'symbol': symbol,
            'prediction': prediction,
            'confidence': f"{confidence:.4f}",
            'close_price': f"{close_price:.2f}",
            'volume': f"{volume:.0f}",
            'balance_simulated': f"{balance_simulated:.2f}",
            'p_and_l': f"{p_and_l:.2f}",
            'accuracy': f"{accuracy:.4f}",
            'win_rate': f"{win_rate:.1f}%",
            'horizon': int(horizon),
            'resolved': 'False',
            'actual_direction': '',
            'is_correct': ''
        }
        
        # Добавить в CSV с выравниванием колонок
        cols = self._expected_columns()
        df = pd.DataFrame([log_data]).reindex(columns=cols)
        df.to_csv(self.csv_log, mode='a', header=False, index=False, na_rep='')
        
        return log_data

    def resolve_predictions(self, symbol, df, horizon=1):
        """Resolve pending predictions based on future candles in df."""
        if not self.csv_log.exists() or df is None or len(df) == 0:
            return 0

        data = pd.read_csv(self.csv_log)
        if len(data) == 0:
            return 0

        if 'resolved' not in data.columns:
            return 0

        # Конвертируем timestamp в datetime
        data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
        
        # Убедимся, что новые колонки есть
        for col in ['exit_price', 'price_change_pct', 'price_change_abs']:
            if col not in data.columns:
                data[col] = ""

        # Фильтруем нерешенные или без расчета изменения
        missing_change = (
            data['exit_price'].isna() | (data['exit_price'] == '') |
            data['price_change_pct'].isna() | (data['price_change_pct'] == '') |
            data['price_change_abs'].isna() | (data['price_change_abs'] == '')
        )

        pending_mask = (data['symbol'] == symbol) & (
            (data['resolved'].isna()) |
            (data['resolved'].astype(str).str.strip() != 'True') |
            (missing_change)
        )
        
        if pending_mask.sum() == 0:
            return 0

        df = df.copy()
        if 'time' not in df.columns:
            return 0
        
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)
        
        # Создаем индекс округленных времен - до ЧАСА, т.к. свечи часовые!
        df['time_rounded'] = df['time'].dt.floor('h')  # Округляем до часа
        
        resolved_count = 0
        
        for idx in data[pending_mask].index:
            ts = data.at[idx, 'timestamp']
            if pd.isna(ts):
                continue
            try:
                row_horizon = int(float(data.at[idx, 'horizon'])) if str(data.at[idx, 'horizon']).strip() != '' else horizon
            except Exception:
                row_horizon = horizon
            
            # Округляем timestamp до ЧАСА для сравнения
            ts_rounded = pd.Timestamp(ts).floor('h')
            
            # Ищем ближайшую свечу (в пределах ±2 часов для надежности)
            time_diff = (df['time_rounded'] - ts_rounded).abs()
            if time_diff.min() > pd.Timedelta(hours=2):
                continue  # Нет подходящей свечи
            
            i = time_diff.idxmin()
            
            # Проверяем, есть ли horizon свечей вперед
            # Если нет, используем последнюю доступную свечу (особенно если это текущая)
            if i + row_horizon >= len(df):
                if i == len(df) - 1:  # Это последняя свеча
                    # Берем последнюю доступную свечу для разрешения
                    i_target = i
                else:
                    continue
            else:
                i_target = i + row_horizon
            
            entry = float(data.at[idx, 'close_price'])
            exit_price = float(df.iloc[i_target]['close'])
            
            # Определяем фактическое направление
            price_change_pct = ((exit_price - entry) / entry) * 100
            
            # Используем небольшой порог (0.05%) чтобы избежать шума
            if price_change_pct > 0.05:
                actual_dir = "UP"
            elif price_change_pct < -0.05:
                actual_dir = "DOWN"
            else:
                actual_dir = "FLAT"  # Нет существенного движения
            
            prediction = str(data.at[idx, 'prediction']).strip()
            
            # Проверяем правильность (FLAT считаем ошибкой)
            is_correct = (prediction == actual_dir) and actual_dir != "FLAT"

            data.loc[idx, 'resolved'] = True
            data.loc[idx, 'actual_direction'] = actual_dir
            data.loc[idx, 'is_correct'] = bool(is_correct)
            data.loc[idx, 'exit_price'] = exit_price
            data.loc[idx, 'price_change_pct'] = price_change_pct
            data.loc[idx, 'price_change_abs'] = exit_price - entry
            resolved_count += 1
            
            
            print(f"    ✓ Разрешен: {prediction} -> {actual_dir} ({price_change_pct:+.2f}%) = {'✅' if is_correct else '❌'}")

        if resolved_count > 0:
            data.to_csv(self.csv_log, index=False)
            print(f"  📊 Разрешено {resolved_count} прогнозов для {symbol}")

        return resolved_count
    
    def log_session_summary(self, symbols, total_predictions, profitable_predictions):
        """Логировать итоги сессии."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_symbols': len(symbols),
            'symbols': ', '.join(symbols),
            'total_predictions': total_predictions,
            'profitable_predictions': profitable_predictions,
            'profitability_rate': f"{(profitable_predictions/total_predictions*100):.1f}%" if total_predictions > 0 else "0%"
        }
        
        summary_file = self.log_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def get_latest_predictions(self, symbol=None, limit=10):
        """Получить последние предсказания."""
        df = pd.read_csv(self.csv_log, on_bad_lines='skip', engine='python')
        
        if symbol:
            df = df[df['symbol'] == symbol]
        
        return df.tail(limit)
    
    def get_statistics(self, symbol=None):
        """Получить статистику по символу или всем."""
        df = pd.read_csv(self.csv_log, on_bad_lines='skip', engine='python')
        
        if symbol:
            df = df[df['symbol'] == symbol]
        
        if len(df) == 0:
            return None
        
        stats = {
            'total_predictions': len(df),
            'symbols': df['symbol'].nunique(),
            'unique_symbols': df['symbol'].unique().tolist(),
            'avg_confidence': df['confidence'].astype(float).mean(),
            'up_predictions': len(df[df['prediction'] == 'UP']),
            'down_predictions': len(df[df['prediction'] == 'DOWN']),
            'unsure_predictions': len(df[df['prediction'] == 'UNSURE'])
        }
        
        return stats
