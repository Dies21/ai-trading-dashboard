def detect_hammer(df):
    """Hammer pattern - бычий разворотный паттерн"""
    df["hammer"] = (
        ((df["high"] - df["close"]) <= (df["close"] - df["open"]) * 0.3)
        & ((df["open"] - df["low"]) >= (df["close"] - df["open"]) * 2)
    )
    return df

def detect_doji(df):
    """Doji pattern - паттерн неопределенности"""
    body = abs(df["close"] - df["open"])
    range_candle = df["high"] - df["low"]
    df["doji"] = (body <= range_candle * 0.1) & (range_candle > 0)
    return df

def detect_shooting_star(df):
    """Shooting Star - медвежий разворотный паттерн"""
    body = abs(df["close"] - df["open"])
    upper_shadow = df["high"] - df[["open", "close"]].max(axis=1)
    lower_shadow = df[["open", "close"]].min(axis=1) - df["low"]
    df["shooting_star"] = (
        (upper_shadow >= body * 2) 
        & (lower_shadow <= body * 0.3)
        & (body > 0)
    )
    return df

def detect_engulfing(df):
    """Bullish/Bearish Engulfing - поглощающий паттерн"""
    # Бычье поглощение
    df["bullish_engulfing"] = (
        (df["close"].shift(1) < df["open"].shift(1))  # предыдущая - медвежья
        & (df["close"] > df["open"])  # текущая - бычья
        & (df["open"] < df["close"].shift(1))  # открытие ниже закрытия предыдущей
        & (df["close"] > df["open"].shift(1))  # закрытие выше открытия предыдущей
    )
    
    # Медвежье поглощение
    df["bearish_engulfing"] = (
        (df["close"].shift(1) > df["open"].shift(1))  # предыдущая - бычья
        & (df["close"] < df["open"])  # текущая - медвежья
        & (df["open"] > df["close"].shift(1))  # открытие выше закрытия предыдущей
        & (df["close"] < df["open"].shift(1))  # закрытие ниже открытия предыдущей
    )
    return df

def detect_morning_star(df):
    """Morning Star - бычий разворотный паттерн из 3 свечей"""
    body1 = abs(df["close"].shift(2) - df["open"].shift(2))
    body2 = abs(df["close"].shift(1) - df["open"].shift(1))
    body3 = abs(df["close"] - df["open"])
    
    df["morning_star"] = (
        (df["close"].shift(2) < df["open"].shift(2))  # 1-я свеча медвежья
        & (body2 < body1 * 0.3)  # 2-я свеча маленькая (звезда)
        & (df["close"] > df["open"])  # 3-я свеча бычья
        & (df["close"] > (df["open"].shift(2) + df["close"].shift(2)) / 2)  # закрытие выше середины 1-й свечи
    )
    return df

def detect_evening_star(df):
    """Evening Star - медвежий разворотный паттерн из 3 свечей"""
    body1 = abs(df["close"].shift(2) - df["open"].shift(2))
    body2 = abs(df["close"].shift(1) - df["open"].shift(1))
    body3 = abs(df["close"] - df["open"])
    
    df["evening_star"] = (
        (df["close"].shift(2) > df["open"].shift(2))  # 1-я свеча бычья
        & (body2 < body1 * 0.3)  # 2-я свеча маленькая (звезда)
        & (df["close"] < df["open"])  # 3-я свеча медвежья
        & (df["close"] < (df["open"].shift(2) + df["close"].shift(2)) / 2)  # закрытие ниже середины 1-й свечи
    )
    return df

def detect_three_white_soldiers(df):
    """Three White Soldiers - три белых солдата (сильный бычий паттерн)"""
    df["three_white_soldiers"] = (
        (df["close"] > df["open"])  # текущая бычья
        & (df["close"].shift(1) > df["open"].shift(1))  # предыдущая бычья
        & (df["close"].shift(2) > df["open"].shift(2))  # 2 назад бычья
        & (df["close"] > df["close"].shift(1))  # каждая выше предыдущей
        & (df["close"].shift(1) > df["close"].shift(2))
        & (df["open"] > df["open"].shift(1))  # открытие в теле предыдущей
        & (df["open"] < df["close"].shift(1))
    )
    return df

def detect_three_black_crows(df):
    """Three Black Crows - три черные вороны (сильный медвежий паттерн)"""
    df["three_black_crows"] = (
        (df["close"] < df["open"])  # текущая медвежья
        & (df["close"].shift(1) < df["open"].shift(1))  # предыдущая медвежья
        & (df["close"].shift(2) < df["open"].shift(2))  # 2 назад медвежья
        & (df["close"] < df["close"].shift(1))  # каждая ниже предыдущей
        & (df["close"].shift(1) < df["close"].shift(2))
        & (df["open"] < df["open"].shift(1))  # открытие в теле предыдущей
        & (df["open"] > df["close"].shift(1))
    )
    return df

def detect_all_patterns(df):
    """Применить все паттерны сразу"""
    df = detect_hammer(df)
    df = detect_doji(df)
    df = detect_shooting_star(df)
    df = detect_engulfing(df)
    df = detect_morning_star(df)
    df = detect_evening_star(df)
    df = detect_three_white_soldiers(df)
    df = detect_three_black_crows(df)
    return df

def calculate_pattern_confidence(df):
    """Рассчитать отдельно уверенность в UP (бычьих паттернах) и DOWN (медвежьих)
    
    Возвращает:
        tuple: (confidence_up, confidence_down)
            confidence_up: сумма бычьих паттернов (0-4)
            confidence_down: сумма медвежьих паттернов (0-4)
    """
    if df.empty:
        return 0, 0
    
    last_row = df.iloc[-1]
    
    # Бычьи паттерны (указывают на рост)
    bullish_patterns = [
        last_row.get('hammer', 0),
        last_row.get('bullish_engulfing', 0),
        last_row.get('morning_star', 0),
        last_row.get('three_white_soldiers', 0),
    ]
    confidence_up = sum(bullish_patterns)
    
    # Медвежьи паттерны (указывают на падение)
    bearish_patterns = [
        last_row.get('shooting_star', 0),
        last_row.get('bearish_engulfing', 0),
        last_row.get('evening_star', 0),
        last_row.get('three_black_crows', 0),
    ]
    confidence_down = sum(bearish_patterns)
    
    return confidence_up, confidence_down
