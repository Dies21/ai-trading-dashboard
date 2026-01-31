import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

FEATURES = ["open","high","low","close","volume","rsi","ema_20","ema_50","macd","atr","bb_position","stoch_k","stoch_d","volume_ma_20","volume_ratio","obv","ad","vroc","rsi_overbought","price_above_bb","macd_negative_divergence","death_cross","volume_divergence"]


def calculate_sample_weights(y, price_changes):
    """Рассчитать веса для образцов на основе стоимости ошибок.
    
    FalsePositive (предсказали UP, а было DOWN) — очень дорого в торговле.
    FalseNegative (предсказали DOWN, а было UP) — упущенная прибыль.
    
    Аргументы:
        y: целевой вектор (0=DOWN, 1=UP)
        price_changes: изменение цены (используется для расчёта штрафов)
    
    Возвращает:
        weights: веса для каждого образца
    """
    weights = np.ones(len(y))
    
    # FalsePositive (предсказание UP когда DOWN) — штраф в 1.5x (дорого в торговле)
    # FalseNegative (предсказание DOWN когда UP) — штраф в 1.2x (упущенная прибыль)
    # TruePositive и TrueNegative — нормальный вес 1.0
    
    for i in range(len(y)):
        if y.iloc[i] == 0:  # Реально было DOWN
            weights[i] = 1.5  # Штраф за FalsePositive выше
        else:  # Реально было UP
            weights[i] = 1.2  # Штраф за FalseNegative меньше
    
    return weights


def calculate_simulated_pnl(y_true, y_pred, price_changes, starting_balance=1000):
    """Рассчитать прибыль/убыток, если бы мы следовали предсказаниям модели.
    
    Аргументы:
        y_true: реальные значения (0=DOWN, 1=UP)
        y_pred: предсказания модели (0=DOWN, 1=UP)
        price_changes: изменение цены за период
        starting_balance: начальный баланс
    
    Возвращает:
        final_balance: финальный баланс
        win_rate: процент прибыльных сделок
        total_trades: количество сделок
    """
    balance = starting_balance
    trades = 0
    wins = 0
    
    # Убедитесь, что используются numpy arrays для индексации
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    price_changes = np.array(price_changes)
    
    for i in range(len(y_true)):
        # Если модель предсказала UP, открываем лонг
        if y_pred[i] == 1:
            trades += 1
            # Если предсказание верно, выигрываем процент от изменения цены
            if y_true[i] == 1 and price_changes[i] > 0:
                profit = balance * (abs(price_changes[i]) / 100)
                balance += profit
                wins += 1
            else:
                # Если предсказание неверно, теряем половину потенциального выигрыша
                loss = balance * (abs(price_changes[i]) / 100) * 0.5
                balance -= loss
    
    if trades == 0:
        return balance, 0, 0
    
    win_rate = (wins / trades * 100)
    
    return balance, win_rate, trades


def train_model(df):
    df = df.copy()
    df["future_close"] = df["close"].shift(-1)
    df["target"] = (df["future_close"] > df["close"]).astype(int)
    
    # Расчитать изменение цены для использования в штрафной системе
    df["price_change_pct"] = ((df["future_close"] - df["close"]) / df["close"] * 100).fillna(0)
    
    df = df.dropna()

    X = df[FEATURES]
    y = df["target"]

    # Нормализация данных (масштабирование признаков)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=FEATURES, index=X.index)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
    
    # Получить соответствующие индексы для выборки тестовых данных
    test_indices = X_test.index
    price_changes_train = df.loc[X_train.index, "price_change_pct"]
    price_changes_test = df.loc[test_indices, "price_change_pct"]
    
    # Рассчитать веса образцов на основе стоимости ошибок
    sample_weights = calculate_sample_weights(y_train, price_changes_train)

    # Быстрая модель с хорошими параметрами (без GridSearchCV для скорости)
    print("Обучение XGBoost с системой штрафов...")
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]),
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
        verbosity=0
    )
    
    # Обучить с весами (штрафы за дорогие ошибки)
    model.fit(X_train, y_train, sample_weight=sample_weights)

    accuracy = model.score(X_test, y_test)
    print(f"Точность на тестовой выборке: {accuracy:.4f}")
    
    # Кросс-валидация для честной оценки
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
    print(f"Кросс-валидация (5 fold): среднее {cv_scores.mean():.4f}, диапазон [{cv_scores.min():.4f}, {cv_scores.max():.4f}]")
    
    # Рассчитать симулированный P&L на тестовой выборке
    y_pred_test = model.predict(X_test)
    
    try:
        final_balance, win_rate, total_trades = calculate_simulated_pnl(
            y_test.values, y_pred_test, price_changes_test.values, starting_balance=1000
        )
        
        print(f"\n💰 Симуляция торговли (начальный баланс: $1000):")
        print(f"  Финальный баланс: ${final_balance:.2f}")
        print(f"  Прибыль/убыток: ${final_balance - 1000:+.2f}")
        print(f"  Win rate: {win_rate:.1f}%")
        print(f"  Количество сделок: {total_trades}")
    except Exception as e:
        print(f"  (P&L расчёт: {e})")
    
    # Сохранить scaler вместе с моделью для дальнейшего использования
    model.scaler = scaler

    return model


def analyze_feature_importance(model, top_n=10):
    """Вывести важность признаков для лучшего понимания модели."""
    feature_importance = model.feature_importances_
    features_with_importance = list(zip(FEATURES, feature_importance))
    features_with_importance.sort(key=lambda x: x[1], reverse=True)
    
    print("\nТоп-важнейшие признаки:")
    for i, (feature, importance) in enumerate(features_with_importance[:top_n], 1):
        print(f"  {i}. {feature}: {importance:.4f}")
    
    return features_with_importance


def evaluate_model(model, df, test_size=0.2):
    """Evaluate trained `model` on the same split as `train_model`.

    Returns a dict with accuracy, precision, recall, f1 and confusion matrix.
    """
    df = df.copy()
    df["future_close"] = df["close"].shift(-1)
    df["target"] = (df["future_close"] > df["close"]).astype(int)
    df = df.dropna()

    X = df[FEATURES]
    y = df["target"]

    # Используем сохранённый scaler для нормализации
    if hasattr(model, 'scaler'):
        X_scaled = model.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=FEATURES, index=X.index)
    else:
        X_scaled = X

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, shuffle=False)

    if X_test.shape[0] == 0:
        return {"error": "no_test_data"}

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    cm = confusion_matrix(y_test, preds)

    print("Оценка на тестовой выборке:")
    print(f"  Точность (accuracy): {acc:.4f}")
    print(f"  Точность предсказаний 'Вверх' (precision): {prec:.4f}")
    print(f"  Полнота для 'Вверх' (recall): {rec:.4f}")
    print(f"  F1 (среднее precision/recall): {f1:.4f}")
    
    # Простое объяснение результатов, без матриц и терминов
    tn, fp, fn, tp = cm.ravel()
    total = int(tn + fp + fn + tp)
    correct = int(tn + tp)
    incorrect = int(total - correct)

    print("Коротко о результатах:")
    print(f"  Всего примеров для теста: {total}")
    print(f"  Правильных предсказаний: {correct} (≈{acc*100:.1f}%)")
    print(f"  Ошибок: {incorrect} (≈{(1-acc)*100:.1f}%)")

    # Объяснение ошибок в простых словах
    predicted_up = int(tp + fp)
    actual_up = int(tp + fn)
    print("")
    print(f"  Модель предсказала 'Вверх' {predicted_up} раз, из них {tp} раз это оказалось верно, {fp} раз — ошибочно.")
    print(f"  Всего на самом деле 'Вверх' было {actual_up} раз, модель пропустила (не заметила) {fn} таких случаев.")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "confusion_matrix": cm}
