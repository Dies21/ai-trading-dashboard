import ccxt
import pandas as pd
import time

class CryptoDataLoader:
    def __init__(self, exchange="binance", symbols=None):
        self.exchange = getattr(ccxt, exchange)()
        # Список активов для анализа
        self.symbols = symbols or ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
        # In-memory storage of market state and candles
        self.market_state = {}  # {symbol: {time, open, high, low, close, volume}}
        self.candles_memory = {}  # {symbol: DataFrame with ranges}
    
    def fetch_ohlcv(self, symbol="BTC/USDT", timeframe="1m", limit=500):
        """Получить OHLCV данные для одного символа и сохранить их в памяти."""
        try:
            data = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(data, columns=["time","open","high","low","close","volume"])
            df["time"] = pd.to_datetime(df["time"], unit="ms")
            df["symbol"] = symbol  # Добавить символ для идентификации

            # Диапазоны свечей
            df["candle_range"] = df["high"] - df["low"]
            df["body_range"] = (df["close"] - df["open"]).abs()
            df["upper_shadow"] = df["high"] - df[["open", "close"]].max(axis=1)
            df["lower_shadow"] = df[["open", "close"]].min(axis=1) - df["low"]

            # Сохранить в памяти все свечи (лимит) и текущее состояние рынка
            self.candles_memory[symbol] = df.copy()
            last = df.iloc[-1]
            self.market_state[symbol] = {
                "time": last["time"],
                "open": float(last["open"]),
                "high": float(last["high"]),
                "low": float(last["low"]),
                "close": float(last["close"]),
                "volume": float(last["volume"]),
                "candle_range": float(last["candle_range"]),
                "body_range": float(last["body_range"])
            }
            return df
        except Exception as e:
            print(f"Ошибка при загрузке {symbol}: {e}")
            return None
    
    def fetch_multiple(self, timeframe="1m", limit=500):
        """Получить данные для всех символов и обновить память."""
        data_dict = {}
        for symbol in self.symbols:
            df = self.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if df is not None:
                data_dict[symbol] = df
        return data_dict

    def get_market_state(self, symbol=None):
        """Получить текущее состояние рынка из памяти."""
        if symbol:
            return self.market_state.get(symbol)
        return self.market_state

    def get_candles_memory(self, symbol=None):
        """Получить все свечи с диапазонами из памяти."""
        if symbol:
            return self.candles_memory.get(symbol)
        return self.candles_memory
    
    def add_symbol(self, symbol):
        """Добавить новый символ для анализа."""
        if symbol not in self.symbols:
            self.symbols.append(symbol)
    
    def remove_symbol(self, symbol):
        """Удалить символ из анализа."""
        if symbol in self.symbols:
            self.symbols.remove(symbol)

if __name__ == "__main__":
    loader = CryptoDataLoader(symbols=["BTC/USDT", "ETH/USDT", "BNB/USDT"])
    data = loader.fetch_multiple()
    for symbol, df in data.items():
        print(f"\n{symbol}:")
        print(df.tail(3))
