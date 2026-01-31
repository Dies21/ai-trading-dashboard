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
    
    def ensure_csv_header(self):
        """Создать заголовок CSV если файл не существует."""
        if not self.csv_log.exists():
            header = pd.DataFrame(columns=[
                'timestamp',
                'symbol',
                'prediction',
                'confidence',
                'close_price',
                'volume',
                'balance_simulated',
                'p_and_l',
                'accuracy',
                'win_rate'
            ])
            header.to_csv(self.csv_log, index=False)
    
    def log_prediction(self, symbol, prediction, confidence, close_price, volume, 
                      balance_simulated, p_and_l, accuracy, win_rate):
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
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'prediction': prediction,
            'confidence': f"{confidence:.4f}",
            'close_price': f"{close_price:.2f}",
            'volume': f"{volume:.0f}",
            'balance_simulated': f"{balance_simulated:.2f}",
            'p_and_l': f"{p_and_l:.2f}",
            'accuracy': f"{accuracy:.4f}",
            'win_rate': f"{win_rate:.1f}%"
        }
        
        # Добавить в CSV
        df = pd.DataFrame([log_data])
        df.to_csv(self.csv_log, mode='a', header=False, index=False)
        
        return log_data
    
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
        df = pd.read_csv(self.csv_log)
        
        if symbol:
            df = df[df['symbol'] == symbol]
        
        return df.tail(limit)
    
    def get_statistics(self, symbol=None):
        """Получить статистику по символу или всем."""
        df = pd.read_csv(self.csv_log)
        
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
