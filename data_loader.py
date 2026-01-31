import ccxt
import pandas as pd
import time

class CryptoDataLoader:
    def __init__(self, exchange="binance", symbols=None):
        self.exchange = getattr(ccxt, exchange)()
        # Список активов для анализа
        self.symbols = symbols or ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
    
    def fetch_ohlcv(self, symbol="BTC/USDT", timeframe="1m", limit=500):
        """Получить OHLCV данные для одного символа."""
        try:
            data = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(data, columns=["time","open","high","low","close","volume"])
            df["time"] = pd.to_datetime(df["time"], unit="ms")
            df["symbol"] = symbol  # Добавить символ для идентификации
            return df
        except Exception as e:
            print(f"Ошибка при загрузке {symbol}: {e}")
            return None
    
    def fetch_multiple(self, timeframe="1m", limit=500):
        """Получить данные для всех символов."""
        data_dict = {}
        for symbol in self.symbols:
            df = self.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if df is not None:
                data_dict[symbol] = df
        return data_dict
    
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
