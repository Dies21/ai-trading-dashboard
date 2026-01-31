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

def detect_harami(df):
    """Bullish/Bearish Harami - внутренний бар"""
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    prev_high = df[["open", "close"]].shift(1).max(axis=1)
    prev_low = df[["open", "close"]].shift(1).min(axis=1)

    # Бычий харами: предыдущая свеча медвежья, текущая маленькая бычья внутри тела
    df["bullish_harami"] = (
        (prev_close < prev_open)
        & (df["close"] > df["open"])
        & (df["open"] >= prev_low)
        & (df["close"] <= prev_high)
    )

    # Медвежий харами: предыдущая свеча бычья, текущая маленькая медвежья внутри тела
    df["bearish_harami"] = (
        (prev_close > prev_open)
        & (df["close"] < df["open"])
        & (df["open"] <= prev_high)
        & (df["close"] >= prev_low)
    )
    return df

def detect_piercing_dark_cloud(df):
    """Piercing Line (bullish) и Dark Cloud Cover (bearish)"""
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    prev_mid = (prev_open + prev_close) / 2

    # Piercing Line: медвежья свеча, затем бычья, закрытие выше середины предыдущей
    df["piercing_line"] = (
        (prev_close < prev_open)
        & (df["open"] < prev_close)
        & (df["close"] > prev_mid)
        & (df["close"] < prev_open)
    )

    # Dark Cloud Cover: бычья свеча, затем медвежья, закрытие ниже середины предыдущей
    df["dark_cloud_cover"] = (
        (prev_close > prev_open)
        & (df["open"] > prev_close)
        & (df["close"] < prev_mid)
        & (df["close"] > prev_open)
    )
    return df

def detect_inverted_hammer_hanging_man(df):
    """Inverted Hammer (bullish) и Hanging Man (bearish)"""
    body = abs(df["close"] - df["open"])
    upper_shadow = df["high"] - df[["open", "close"]].max(axis=1)
    lower_shadow = df[["open", "close"]].min(axis=1) - df["low"]

    # Inverted Hammer: длинная верхняя тень, маленькое тело, почти нет нижней
    df["inverted_hammer"] = (
        (upper_shadow >= body * 2)
        & (lower_shadow <= body * 0.3)
        & (body > 0)
        & (df["close"] > df["open"])
    )

    # Hanging Man: длинная нижняя тень, маленькое тело, почти нет верхней
    df["hanging_man"] = (
        (lower_shadow >= body * 2)
        & (upper_shadow <= body * 0.3)
        & (body > 0)
        & (df["close"] < df["open"])
    )
    return df

def detect_tweezer(df):
    """Tweezer Top/Bottom"""
    prev_high = df["high"].shift(1)
    prev_low = df["low"].shift(1)
    tolerance = (df["high"] - df["low"]) * 0.1

    df["tweezer_top"] = (
        (abs(df["high"] - prev_high) <= tolerance)
        & (df["close"].shift(1) > df["open"].shift(1))
        & (df["close"] < df["open"])
    )

    df["tweezer_bottom"] = (
        (abs(df["low"] - prev_low) <= tolerance)
        & (df["close"].shift(1) < df["open"].shift(1))
        & (df["close"] > df["open"])
    )
    return df

def detect_marubozu(df):
    """Bullish/Bearish Marubozu"""
    body = abs(df["close"] - df["open"])
    range_candle = df["high"] - df["low"]
    small_shadow = range_candle * 0.05

    df["bullish_marubozu"] = (
        (df["close"] > df["open"])
        & ((df["high"] - df["close"]) <= small_shadow)
        & ((df["open"] - df["low"]) <= small_shadow)
        & (body > 0)
    )

    df["bearish_marubozu"] = (
        (df["close"] < df["open"])
        & ((df["high"] - df["open"]) <= small_shadow)
        & ((df["close"] - df["low"]) <= small_shadow)
        & (body > 0)
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
    df = detect_harami(df)
    df = detect_piercing_dark_cloud(df)
    df = detect_inverted_hammer_hanging_man(df)
    df = detect_tweezer(df)
    df = detect_marubozu(df)
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
        last_row.get('inverted_hammer', 0),
        last_row.get('bullish_engulfing', 0),
        last_row.get('bullish_harami', 0),
        last_row.get('piercing_line', 0),
        last_row.get('tweezer_bottom', 0),
        last_row.get('bullish_marubozu', 0),
        last_row.get('morning_star', 0),
        last_row.get('three_white_soldiers', 0),
    ]
    confidence_up = sum(bullish_patterns)
    
    # Медвежьи паттерны (указывают на падение)
    bearish_patterns = [
        last_row.get('shooting_star', 0),
        last_row.get('hanging_man', 0),
        last_row.get('bearish_engulfing', 0),
        last_row.get('bearish_harami', 0),
        last_row.get('dark_cloud_cover', 0),
        last_row.get('tweezer_top', 0),
        last_row.get('bearish_marubozu', 0),
        last_row.get('evening_star', 0),
        last_row.get('three_black_crows', 0),
    ]
    confidence_down = sum(bearish_patterns)
    
    return confidence_up, confidence_down
