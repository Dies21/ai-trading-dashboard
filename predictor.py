def predict_next(model, df, up_threshold=0.50, down_threshold=0.45):
    """Предсказать следующее движение с учётом уверенности модели.
    
    Аргументы:
        model: Обученная модель
        df: DataFrame с данными
        up_threshold: Минимальная уверенность для UP предсказания (0-1)
        down_threshold: Минимальная уверенность для DOWN предсказания (0-1)
    
    Возвращает:
        tuple: (prediction, confidence, prob_down, prob_up, reliability)
            prediction: 'UP' (Вверх), 'DOWN' (Вниз), или 'UNSURE' (Неуверен)
            confidence: уверенность модели (0-1)
            prob_down: вероятность DOWN
            prob_up: вероятность UP
            reliability: надёжность ('HIGH', 'MEDIUM', 'LOW')
    """
    from model import FEATURES
    if df.empty:
        return "NO_DATA", 0.0, 0.0, 0.0, "NONE"
    
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
    
    # Определить надёжность прогноза
    max_prob = max(prob_up, prob_down)
    if max_prob >= 0.60:
        reliability = "HIGH"    # Высокая надёжность
    elif max_prob >= 0.50:
        reliability = "MEDIUM"  # Средняя надёжность
    else:
        reliability = "LOW"     # Низкая надёжность
    
    # Разные пороги для UP и DOWN для более сбалансированных сигналов
    if prob_down > prob_up and prob_down >= down_threshold:
        return "DOWN", prob_down, prob_down, prob_up, reliability
    elif prob_up > prob_down and prob_up >= up_threshold:
        return "UP", prob_up, prob_down, prob_up, reliability
    else:
        return "UNSURE", max(prob_up, prob_down), prob_down, prob_up, reliability
