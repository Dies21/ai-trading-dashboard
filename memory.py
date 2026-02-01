"""Memory storage for market state and candle history."""
import pandas as pd
from datetime import datetime
from pathlib import Path
import json


class MarketMemory:
    """Хранилище текущего состояния рынка для каждого актива."""
    
    def __init__(self, storage_file="logs/market_state.json"):
        self.storage_file = Path(storage_file)
        self.storage_file.parent.mkdir(exist_ok=True)
        self.state = {}  # {symbol: {time, open, high, low, close, volume, ranges}}
        self.load()
    
    def update(self, symbol, time, open_price, high, low, close, volume, 
               candle_range=None, body_range=None):
        """Обновить состояние рынка для символа."""
        self.state[symbol] = {
            "time": str(time),
            "open": float(open_price),
            "high": float(high),
            "low": float(low),
            "close": float(close),
            "volume": float(volume),
            "candle_range": float(candle_range) if candle_range else float(high - low),
            "body_range": float(body_range) if body_range else float(abs(close - open_price)),
            "last_update": datetime.now().isoformat()
        }
        self.save()
    
    def get(self, symbol):
        """Получить состояние рынка для символа."""
        return self.state.get(symbol)
    
    def get_all(self):
        """Получить состояния всех рынков."""
        return self.state
    
    def get_price(self, symbol):
        """Получить текущую цену закрытия."""
        state = self.get(symbol)
        return state["close"] if state else None
    
    def get_volume(self, symbol):
        """Получить текущий объём."""
        state = self.get(symbol)
        return state["volume"] if state else None
    
    def save(self):
        """Сохранить в файл."""
        try:
            with open(self.storage_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save market state: {e}")
    
    def load(self):
        """Загрузить из файла."""
        try:
            if self.storage_file.exists():
                with open(self.storage_file, 'r') as f:
                    self.state = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load market state: {e}")
            self.state = {}


class CandleMemory:
    """Хранилище истории свечей с диапазонами для каждого актива."""
    
    def __init__(self, storage_dir="logs/candles"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.memory = {}  # {symbol: DataFrame}
        self.max_candles = 1000  # Максимум свечей в памяти
    
    def update(self, symbol, df):
        """Обновить историю свечей для символа.
        
        Args:
            symbol: Торговая пара (например, BTC/USDT)
            df: DataFrame с колонками [time, open, high, low, close, volume, 
                candle_range, body_range, upper_shadow, lower_shadow]
        """
        # Ограничить размер истории
        if len(df) > self.max_candles:
            df = df.tail(self.max_candles)
        
        self.memory[symbol] = df.copy()
        self.save(symbol)
    
    def get(self, symbol, last_n=None):
        """Получить историю свечей для символа.
        
        Args:
            symbol: Торговая пара
            last_n: Количество последних свечей (None = все)
        """
        df = self.memory.get(symbol)
        if df is None:
            df = self.load(symbol)
        
        if df is not None and last_n:
            return df.tail(last_n)
        return df
    
    def get_last_candle(self, symbol):
        """Получить последнюю свечу."""
        df = self.get(symbol, last_n=1)
        if df is not None and len(df) > 0:
            return df.iloc[-1].to_dict()
        return None
    
    def get_range_stats(self, symbol, last_n=50):
        """Получить статистику диапазонов за последние N свечей."""
        df = self.get(symbol, last_n=last_n)
        if df is None or len(df) == 0:
            return None
        
        return {
            "avg_candle_range": df["candle_range"].mean(),
            "max_candle_range": df["candle_range"].max(),
            "min_candle_range": df["candle_range"].min(),
            "avg_body_range": df["body_range"].mean(),
            "avg_upper_shadow": df["upper_shadow"].mean(),
            "avg_lower_shadow": df["lower_shadow"].mean(),
            "volatility": df["candle_range"].std()
        }
    
    def get_all_symbols(self):
        """Получить список всех символов в памяти."""
        return list(self.memory.keys())
    
    def save(self, symbol):
        """Сохранить свечи символа в файл."""
        try:
            df = self.memory.get(symbol)
            if df is not None:
                # Безопасное имя файла (заменить / на _)
                safe_symbol = symbol.replace("/", "_")
                file_path = self.storage_dir / f"{safe_symbol}.parquet"
                df.to_parquet(file_path)
        except Exception as e:
            print(f"Warning: Could not save candles for {symbol}: {e}")
    
    def load(self, symbol):
        """Загрузить свечи символа из файла."""
        try:
            safe_symbol = symbol.replace("/", "_")
            file_path = self.storage_dir / f"{safe_symbol}.parquet"
            if file_path.exists():
                df = pd.read_parquet(file_path)
                self.memory[symbol] = df
                return df
        except Exception as e:
            print(f"Warning: Could not load candles for {symbol}: {e}")
        return None
    
    def clear(self, symbol=None):
        """Очистить память (все или конкретный символ)."""
        if symbol:
            if symbol in self.memory:
                del self.memory[symbol]
        else:
            self.memory = {}


if __name__ == "__main__":
    # Тест
    market = MarketMemory()
    market.update("BTC/USDT", datetime.now(), 50000, 51000, 49500, 50500, 1000000)
    print("Market state:", market.get("BTC/USDT"))
    
    candles = CandleMemory()
    test_df = pd.DataFrame({
        "time": pd.date_range(start="2024-01-01", periods=10, freq="1H"),
        "open": [50000, 50100, 50200, 50300, 50400, 50500, 50600, 50700, 50800, 50900],
        "high": [51000, 51100, 51200, 51300, 51400, 51500, 51600, 51700, 51800, 51900],
        "low": [49500, 49600, 49700, 49800, 49900, 50000, 50100, 50200, 50300, 50400],
        "close": [50500, 50600, 50700, 50800, 50900, 51000, 51100, 51200, 51300, 51400],
        "volume": [1000000] * 10,
        "candle_range": [1500] * 10,
        "body_range": [500] * 10,
        "upper_shadow": [500] * 10,
        "lower_shadow": [500] * 10
    })
    candles.update("BTC/USDT", test_df)
    print("Last candle:", candles.get_last_candle("BTC/USDT"))
    print("Range stats:", candles.get_range_stats("BTC/USDT"))
