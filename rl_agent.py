import numpy as np

ACTIONS = ["HOLD", "BUY", "SELL"]


class RLAgent:
    """Lightweight Q-learning agent with a simple trading environment.

    Discretizes a few technical indicators into bins and learns a Q-table.
    Includes basic risk controls: fees, stop-loss, take-profit, max holding, drawdown penalty.
    """

    def __init__(
        self,
        alpha=0.1,
        gamma=0.9,
        epsilon=0.1,
        fee_rate=0.0004,
        stop_loss_pct=0.01,
        take_profit_pct=0.02,
        max_holding=20,
        drawdown_penalty=0.1
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.fee_rate = fee_rate
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_holding = max_holding
        self.drawdown_penalty = drawdown_penalty
        self.q_table = {}

        # Bins for discretization
        self.rsi_bins = [30, 50, 70]
        self.bb_bins = [0.2, 0.5, 0.8]
        self.stoch_bins = [20, 50, 80]
        self.vol_ratio_bins = [0.8, 1.2]

    def _bin(self, value, bins):
        for i, b in enumerate(bins):
            if value <= b:
                return i
        return len(bins)

    def _state(self, row):
        rsi = self._bin(row["rsi"], self.rsi_bins)
        macd_sign = 0 if row["macd"] == 0 else (1 if row["macd"] > 0 else -1)
        bb_pos = self._bin(row["bb_position"], self.bb_bins)
        stoch_k = self._bin(row["stoch_k"], self.stoch_bins)
        vol_ratio = self._bin(row["volume_ratio"], self.vol_ratio_bins)
        return (rsi, macd_sign, bb_pos, stoch_k, vol_ratio)

    def _ensure_state(self, state):
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in ACTIONS}

    def _reward(self, action, next_return):
        if action == "BUY":
            return float(next_return)
        if action == "SELL":
            return float(-next_return)
        return 0.0

    def train(self, df):
        """Train Q-table from historical data in `df` with simple trade simulation."""
        required = ["close", "high", "low", "rsi", "macd", "bb_position", "stoch_k", "volume_ratio"]
        if not all(col in df.columns for col in required):
            raise ValueError("RLAgent requires: close, high, low, rsi, macd, bb_position, stoch_k, volume_ratio")

        data = df.dropna(subset=required).copy()
        if len(data) < 3:
            return

        position = 0
        entry_price = None
        holding = 0
        equity = 1.0
        peak_equity = 1.0

        for i in range(len(data) - 1):
            row = data.iloc[i]
            next_row = data.iloc[i + 1]

            state = self._state(row)
            next_state = self._state(next_row)

            self._ensure_state(state)
            self._ensure_state(next_state)

            best_next = max(self.q_table[next_state].values())

            if np.random.rand() < self.epsilon:
                action = np.random.choice(ACTIONS)
            else:
                action = max(self.q_table[state], key=self.q_table[state].get)

            reward = 0.0
            if position == 0:
                if action in ["BUY", "SELL"]:
                    position = 1 if action == "BUY" else -1
                    entry_price = row["close"]
                    holding = 0
                    equity *= (1 - self.fee_rate)
            else:
                holding += 1
                exit_price = None

                if position == 1:
                    sl_price = entry_price * (1 - self.stop_loss_pct)
                    tp_price = entry_price * (1 + self.take_profit_pct)
                    if next_row["low"] <= sl_price:
                        exit_price = sl_price
                    elif next_row["high"] >= tp_price:
                        exit_price = tp_price
                else:
                    sl_price = entry_price * (1 + self.stop_loss_pct)
                    tp_price = entry_price * (1 - self.take_profit_pct)
                    if next_row["high"] >= sl_price:
                        exit_price = sl_price
                    elif next_row["low"] <= tp_price:
                        exit_price = tp_price

                if exit_price is None and ((position == 1 and action == "SELL") or (position == -1 and action == "BUY")):
                    exit_price = next_row["close"]

                if exit_price is None and holding >= self.max_holding:
                    exit_price = next_row["close"]

                if exit_price is not None:
                    trade_return = ((exit_price - entry_price) / entry_price) * position
                    equity *= (1 + trade_return)
                    equity *= (1 - self.fee_rate)
                    reward += trade_return
                    position = 0
                    entry_price = None
                    holding = 0

            if equity > peak_equity:
                peak_equity = equity
            drawdown = (peak_equity - equity) / max(peak_equity, 1e-9)
            reward -= drawdown * self.drawdown_penalty

            old_q = self.q_table[state][action]
            new_q = old_q + self.alpha * (reward + self.gamma * best_next - old_q)
            self.q_table[state][action] = new_q

    def act(self, df):
        """Return action and a confidence score for the latest row."""
        required = ["rsi", "macd", "bb_position", "stoch_k", "volume_ratio"]
        if not all(col in df.columns for col in required):
            return "HOLD", 0.0

        last = df.dropna(subset=required).iloc[-1]
        state = self._state(last)
        self._ensure_state(state)

        q_vals = np.array([self.q_table[state][a] for a in ACTIONS])
        # Softmax confidence
        exp = np.exp(q_vals - np.max(q_vals))
        probs = exp / (exp.sum() + 1e-9)
        action = ACTIONS[int(np.argmax(probs))]
        confidence = float(np.max(probs))
        return action, confidence
