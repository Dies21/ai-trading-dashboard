import numpy as np
import pandas as pd
from data_loader import CryptoDataLoader
from features import add_indicators
from patterns import detect_all_patterns
from model import train_model
from predictor import predict_next


def backtest_symbol(
    df,
    horizon=3,
    train_size=300,
    test_size=200,
    step=100,
    up_threshold=0.5,
    down_threshold=0.4,
    fee_rate=0.0004,
    slippage=0.0005
):
    """Honest walk-forward backtest with fees and slippage."""
    df = df.copy().reset_index(drop=True)
    if len(df) < train_size + test_size + horizon:
        return {"error": "not_enough_data"}

    results = []
    start = 0

    while start + train_size + test_size + horizon <= len(df):
        train_df = df.iloc[start:start + train_size].copy()
        test_df = df.iloc[start + train_size:start + train_size + test_size].copy()

        model = train_model(train_df, do_cv=False, horizon=horizon)
        if model is None:
            start += step
            continue

        trades = 0
        wins = 0
        pnl = 0.0
        correct = 0
        total_preds = 0

        for i in range(len(test_df) - horizon):
            idx = start + train_size + i
            slice_df = df.iloc[:idx + 1].copy()

            prediction, confidence, prob_down, prob_up, reliability, pattern_up, pattern_down = predict_next(
                model,
                slice_df,
                up_threshold=up_threshold,
                down_threshold=down_threshold
            )

            if prediction == "UNSURE":
                continue

            entry = df.iloc[idx]["close"]
            exit_price = df.iloc[idx + horizon]["close"]
            raw_return = (exit_price - entry) / entry

            if prediction == "DOWN":
                raw_return = -raw_return

            # Fees: entry + exit
            trade_return = raw_return - (fee_rate * 2) - (slippage * 2)
            pnl += trade_return
            trades += 1

            actual_up = exit_price > entry
            if (prediction == "UP" and actual_up) or (prediction == "DOWN" and not actual_up):
                wins += 1
                correct += 1
            total_preds += 1

        if trades > 0:
            results.append({
                "trades": trades,
                "win_rate": wins / trades,
                "avg_return": pnl / trades,
                "pnl": pnl,
                "accuracy": correct / total_preds if total_preds > 0 else 0.0
            })

        start += step

    if not results:
        return {"error": "no_trades"}

    avg_accuracy = float(np.mean([r["accuracy"] for r in results]))
    avg_win_rate = float(np.mean([r["win_rate"] for r in results]))
    avg_return = float(np.mean([r["avg_return"] for r in results]))
    total_pnl = float(np.sum([r["pnl"] for r in results]))

    return {
        "splits": len(results),
        "avg_accuracy": avg_accuracy,
        "avg_win_rate": avg_win_rate,
        "avg_return": avg_return,
        "total_pnl": total_pnl
    }


def optimize_thresholds(
    df,
    horizon=3,
    train_size=300,
    test_size=200,
    step=100,
    fee_rate=0.0004,
    slippage=0.0005
):
    """Simple grid search for thresholds to maximize accuracy and win rate."""
    best = None
    for up_th in [0.45, 0.50, 0.55, 0.60]:
        for down_th in [0.30, 0.35, 0.40, 0.45]:
            metrics = backtest_symbol(
                df,
                horizon=horizon,
                train_size=train_size,
                test_size=test_size,
                step=step,
                up_threshold=up_th,
                down_threshold=down_th,
                fee_rate=fee_rate,
                slippage=slippage
            )
            if "error" in metrics:
                continue
            score = (metrics["avg_accuracy"] * 0.6) + (metrics["avg_win_rate"] * 0.4)
            if best is None or score > best["score"]:
                best = {"score": score, "up": up_th, "down": down_th, "metrics": metrics}
    return best or {"error": "no_valid_results"}


if __name__ == "__main__":
    loader = CryptoDataLoader(symbols=["BTC/USDT"])
    df = loader.fetch_ohlcv("BTC/USDT", timeframe="1h", limit=1000)
    df = add_indicators(df)
    df = detect_all_patterns(df)

    best = optimize_thresholds(df)
    print("Best thresholds:", best)
