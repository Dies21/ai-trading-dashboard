class RiskManager:
    """Simple risk manager for signal gating and position sizing."""

    def __init__(
        self,
        balance=1000.0,
        risk_per_trade=0.01,
        stop_loss_pct=0.012,
        take_profit_pct=0.02,
        max_drawdown_pct=0.2,
        min_confidence=0.55,
        max_atr_pct=0.035,
        cooldown_bars=0
    ):
        self.balance = float(balance)
        self.peak_balance = float(balance)
        self.risk_per_trade = float(risk_per_trade)
        self.stop_loss_pct = float(stop_loss_pct)
        self.take_profit_pct = float(take_profit_pct)
        self.max_drawdown_pct = float(max_drawdown_pct)
        self.min_confidence = float(min_confidence)
        self.max_atr_pct = float(max_atr_pct)
        self.cooldown_bars = int(cooldown_bars)
        self._cooldown_left = 0

    def evaluate_trade(self, confidence, atr, price):
        """Return trade gating and sizing based on risk constraints."""
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            return {
                "allowed": False,
                "reason": "cooldown",
                "position_size": 0.0,
                "stop_loss": None,
                "take_profit": None
            }

        if confidence < self.min_confidence:
            return {
                "allowed": False,
                "reason": "low_confidence",
                "position_size": 0.0,
                "stop_loss": None,
                "take_profit": None
            }

        if price <= 0:
            return {
                "allowed": False,
                "reason": "invalid_price",
                "position_size": 0.0,
                "stop_loss": None,
                "take_profit": None
            }

        atr_pct = (atr / price) if atr is not None else 0.0
        if atr_pct > self.max_atr_pct:
            return {
                "allowed": False,
                "reason": "high_volatility",
                "position_size": 0.0,
                "stop_loss": None,
                "take_profit": None
            }

        drawdown = (self.peak_balance - self.balance) / max(self.peak_balance, 1e-9)
        if drawdown >= self.max_drawdown_pct:
            return {
                "allowed": False,
                "reason": "max_drawdown",
                "position_size": 0.0,
                "stop_loss": None,
                "take_profit": None
            }

        risk_amount = self.balance * self.risk_per_trade
        position_size = risk_amount / max(self.stop_loss_pct * price, 1e-9)

        return {
            "allowed": True,
            "reason": "ok",
            "position_size": float(position_size),
            "stop_loss": float(price * (1 - self.stop_loss_pct)),
            "take_profit": float(price * (1 + self.take_profit_pct))
        }

    def update_after_trade(self, pnl):
        """Update balance and drawdown after a trade result."""
        self.balance += float(pnl)
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        if pnl < 0:
            self._cooldown_left = max(self._cooldown_left, 1)
