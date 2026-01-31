import ta

def add_indicators(df):
    # Существующие индикаторы
    df["rsi"] = ta.momentum.rsi(df["close"])
    df["ema_20"] = ta.trend.ema_indicator(df["close"], window=20)
    df["ema_50"] = ta.trend.ema_indicator(df["close"], window=50)
    df["macd"] = ta.trend.macd_diff(df["close"])
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"])
    
    # Новые индикаторы: Bollinger Bands
    bb_high = ta.volatility.bollinger_hband(df["close"], window=20)
    bb_low = ta.volatility.bollinger_lband(df["close"], window=20)
    bb_mid = ta.volatility.bollinger_mavg(df["close"], window=20)
    df["bb_position"] = (df["close"] - bb_low) / (bb_high - bb_low + 1e-6)  # Позиция цены в полосе
    
    # Новые индикаторы: Stochastic Oscillator
    df["stoch_k"] = ta.momentum.stoch(df["high"], df["low"], df["close"], window=14)
    df["stoch_d"] = ta.momentum.stoch_signal(df["high"], df["low"], df["close"], window=14)
    
    # Новые Volume-based индикаторы
    # Volume Moving Average (20-период)
    df["volume_ma_20"] = df["volume"].rolling(window=20).mean()
    # Volume Ratio (текущий volume / средний volume за последние 20 периодов)
    df["volume_ratio"] = df["volume"] / (df["volume_ma_20"] + 1e-6)
    
    # On-Balance Volume (OBV) - накопительный индикатор объёма
    df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"])
    
    # Accumulation/Distribution Line (A/D)
    df["ad"] = ta.volume.acc_dist_index(df["high"], df["low"], df["close"], df["volume"])
    
    # Volume Rate of Change (VROC)
    df["vroc"] = ta.volume.volume_price_trend(df["close"], df["volume"])
    
    # Медвежі сигнали (bearish indicators) для кращого прогнозування падіння
    # 1. Перевищення в перекупленій зоні RSI (RSI > 70 - можлива корекція вниз)
    df["rsi_overbought"] = (df["rsi"] > 70).astype(int)
    
    # 2. Ціна вище верхньої полоси Боллінджера (можливе повернення до середньої)
    df["price_above_bb"] = (df["close"] > bb_high).astype(int)
    
    # 3. Негативна дивергенція MACD (ціна зростає, але MACD падає)
    df["macd_negative_divergence"] = ((df["close"].diff() > 0) & (df["macd"].diff() < 0)).astype(int)
    
    # 4. "Death Cross" - швидка EMA перетинає повільну EMA зверху вниз
    df["death_cross"] = ((df["ema_20"] < df["ema_50"]) & (df["ema_20"].shift(1) >= df["ema_50"].shift(1))).astype(int)
    
    # 5. Падіння обсягу при зростанні ціни (слабкий зростання - можливий розворот)
    df["volume_divergence"] = ((df["close"].diff() > 0) & (df["volume"].diff() < 0)).astype(int)
    
    return df.fillna(0)
