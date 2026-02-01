import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Простые эффективные признаки
FEATURES = ["close", "volume", "rsi", "ema_20", "ema_50", "macd", "atr", "bb_position"]


def calculate_simulated_pnl(y_true, y_pred, price_changes, starting_balance=1000):
    """Симуляция торговли по сигналам модели."""
    balance = starting_balance
    trades = 0
    wins = 0
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    price_changes = np.array(price_changes)
    
    for i in range(len(y_true)):
        if y_pred[i] == 1:  # Предсказание UP
            trades += 1
            if y_true[i] == 1 and price_changes[i] > 0:
                profit = balance * (abs(price_changes[i]) / 100)
                balance += profit
                wins += 1
            else:
                loss = balance * (abs(price_changes[i]) / 100) * 0.5
                balance -= loss
    
    if trades == 0:
        return balance, 0, 0
    
    win_rate = (wins / trades * 100)
    return balance, win_rate, trades


def train_model(df):
    """Упрощенное быстрое обучение модели."""
    df = df.copy()
    df["future_close"] = df["close"].shift(-1)
    df["target"] = (df["future_close"] > df["close"]).astype(int)
    df["price_change_pct"] = ((df["future_close"] - df["close"]) / df["close"] * 100).fillna(0)
    df = df.dropna()

    X = df[FEATURES]
    y = df["target"]

    # Быстрая нормализация
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=FEATURES, index=X.index)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
    
    test_indices = X_test.index
    price_changes_test = df.loc[test_indices, "price_change_pct"]

    # Простая быстрая модель
    print("Обучение XGBoost (упрощенная версия)...")
    model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Точность: {accuracy:.4f}")
    
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


def analyze_feature_importance(model, top_n=8):
    """Важность признаков."""
    feature_importance = model.feature_importances_
    features_with_importance = list(zip(FEATURES, feature_importance))
    features_with_importance.sort(key=lambda x: x[1], reverse=True)
    
    print("\nВажность признаков:")
    for i, (feature, importance) in enumerate(features_with_importance[:top_n], 1):
        print(f"  {i}. {feature}: {importance:.4f}")
    
    return features_with_importance


def evaluate_model(model, df, test_size=0.2):
    """Простая оценка модели."""
    df = df.copy()
    df["future_close"] = df["close"].shift(-1)
    df["target"] = (df["future_close"] > df["close"]).astype(int)
    df = df.dropna()

    X = df[FEATURES]
    y = df["target"]

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

    print(f"Точность: {acc:.4f}")
    return {"accuracy": acc}
