def detect_hammer(df):
    df["hammer"] = (
        ((df["high"] - df["close"]) <= (df["close"] - df["open"]) * 0.3)
        & ((df["open"] - df["low"]) >= (df["close"] - df["open"]) * 2)
    )
    return df
