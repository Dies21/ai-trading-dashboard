def predict_next(model, df, up_threshold=0.48, down_threshold=0.35):
    """Предсказать следующее движение с учётом уверенности модели.
    Оптимизирован для баланса DOWN/UP сигналов.
    
    Аргументы:
        model: Обученная модель
        df: DataFrame с данными
        up_threshold: Порог для UP (0-1)
        down_threshold: Порог для DOWN (0-1)
    
    Возвращает:
        tuple: (prediction, confidence, prob_down, prob_up, reliability)
    """
    from model import FEATURES
    if df.empty:
        return "NO_DATA", 0.0, 0.0, 0.0, "NONE"
    
    last_row = df[FEATURES].iloc[-1:]
    if hasattr(model, 'scaler'):
        last_row_scaled = model.scaler.transform(last_row)
    else:
        last_row_scaled = last_row
    
    proba = model.predict_proba(last_row_scaled)[0]
    prob_down = proba[0]
    prob_up = proba[1]
    
    max_prob = max(prob_up, prob_down)
    if max_prob >= 0.55:
        reliability = "HIGH"
    elif max_prob >= 0.45:
        reliability = "MEDIUM"
    else:
        reliability = "LOW"
    
    # Агресивна логіка: DOWN отримує перевагу
    # Якщо DOWN більший И більший за поріг - это DOWN
    # Якщо різниця мала (< 0.05) - також може бути DOWN
    if prob_down >= down_threshold:
        if prob_down > prob_up or (prob_up - prob_down < 0.05 and prob_down >= down_threshold):
            return "DOWN", prob_down, prob_down, prob_up, reliability
    
    if prob_up >= up_threshold and prob_up > prob_down:
        return "UP", prob_up, prob_down, prob_up, reliability
    
    return "UNSURE", max(prob_up, prob_down), prob_down, prob_up, reliability
