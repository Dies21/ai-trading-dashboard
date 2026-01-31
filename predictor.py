def predict_next(model, df, confidence_threshold=0.55):
    """Предсказать следующее движение с учётом уверенности модели.
    
    Аргументы:
        model: Обученная модель
        df: DataFrame с данными
        confidence_threshold: Минимальная уверенность для предсказания (0-1)
    
    Возвращает:
        tuple: (prediction, confidence)
            prediction: 'UP' (Вверх), 'DOWN' (Вниз), или 'UNSURE' (Неуверен)
            confidence: уверенность модели (0-1)
    """
    from model import FEATURES
    if df.empty:
        return "NO_DATA", 0.0
    
    # Получить последний ряд и нормализовать его через scaler модели
    last_row = df[FEATURES].iloc[-1:]
    
    if hasattr(model, 'scaler'):
        last_row_scaled = model.scaler.transform(last_row)
    else:
        last_row_scaled = last_row
    
    # Получить вероятность предсказания
    proba = model.predict_proba(last_row_scaled)[0]  # [prob_DOWN, prob_UP]
    prob_up = proba[1]  # Вероятность класса "UP"
    prob_down = proba[0]  # Вероятность класса "DOWN"
    
    # Предсказывать только если модель уверена
    max_confidence = max(prob_up, prob_down)
    
    if max_confidence < confidence_threshold:
        return "UNSURE", max_confidence
    
    prediction = "UP" if prob_up > prob_down else "DOWN"
    return prediction, max_confidence
