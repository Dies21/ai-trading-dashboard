def predict_next(model, df, up_threshold=0.48, down_threshold=0.35):
    """Предсказать следующее движение с учётом уверенности модели и паттернов.
    Оптимизирован для баланса DOWN/UP сигналов.
    
    Аргументы:
        model: Обученная модель
        df: DataFrame с данными
        up_threshold: Порог для UP (0-1)
        down_threshold: Порог для DOWN (0-1)
    
    Возвращает:
        tuple: (prediction, confidence, prob_down, prob_up, reliability, pattern_up, pattern_down)
    """
    from model import FEATURES
    from patterns import calculate_pattern_confidence
    
    if df.empty:
        return "NO_DATA", 0.0, 0.0, 0.0, "NONE", 0, 0
    
    last_row = df[FEATURES].iloc[-1:]
    if hasattr(model, 'scaler'):
        last_row_scaled = model.scaler.transform(last_row)
    else:
        last_row_scaled = last_row
    
    proba = model.predict_proba(last_row_scaled)[0]
    prob_down = proba[0]
    prob_up = proba[1]
    
    # Получить уверенность по паттернам
    pattern_up, pattern_down = calculate_pattern_confidence(df)
    
    max_prob = max(prob_up, prob_down)
    if max_prob >= 0.55:
        reliability = "HIGH"
    elif max_prob >= 0.45:
        reliability = "MEDIUM"
    else:
        reliability = "LOW"
    
    # Добавить влияние паттернов (каждый паттерн = +0.08 к уверенности)
    adjusted_prob_up = prob_up + (pattern_up * 0.08)
    adjusted_prob_down = prob_down + (pattern_down * 0.08)
    
    # Нормализировать если превышены 1.0
    total = adjusted_prob_up + adjusted_prob_down
    if total > 1.0:
        adjusted_prob_up = adjusted_prob_up / total
        adjusted_prob_down = adjusted_prob_down / total
    
    # Агресивна логіка: DOWN отримує перевагу
    if adjusted_prob_down >= down_threshold:
        if adjusted_prob_down > adjusted_prob_up or (adjusted_prob_up - adjusted_prob_down < 0.05 and adjusted_prob_down >= down_threshold):
            return "DOWN", adjusted_prob_down, adjusted_prob_down, adjusted_prob_up, reliability, pattern_up, pattern_down
    
    if adjusted_prob_up >= up_threshold and adjusted_prob_up > adjusted_prob_down:
        return "UP", adjusted_prob_up, adjusted_prob_down, adjusted_prob_up, reliability, pattern_up, pattern_down
    
    return "UNSURE", max(adjusted_prob_up, adjusted_prob_down), adjusted_prob_down, adjusted_prob_up, reliability, pattern_up, pattern_down
